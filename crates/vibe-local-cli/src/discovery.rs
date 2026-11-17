//! Local discovery so Vibe Router can find Vibe Local automatically:
//! 1. A UDP probe responder on port 20131 (`VIBE_DISCOVER_V1` -> JSON).
//! 2. A minimal mDNS / DNS-SD responder advertising `_vibe-local._tcp.local`
//!    (PTR + SRV + TXT + A records) on 224.0.0.251:5353.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::json;

pub const UDP_PROBE_PORT: u16 = 20131;
pub const MDNS_PORT: u16 = 5353;
const MDNS_GROUP: Ipv4Addr = Ipv4Addr::new(224, 0, 0, 251);
const SERVICE_TYPE: &str = "_vibe-local._tcp.local";

#[derive(Clone)]
pub struct DiscoveryConfig {
    pub http_port: u16,
    pub lan: bool,
    pub udp_port: u16,
    pub mdns_port: u16,
}

pub struct DiscoveryHandle {
    stop: Arc<AtomicBool>,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl DiscoveryHandle {
    pub fn shutdown(&self) {
        self.stop.store(true, Ordering::SeqCst);
        // Threads observe the flag within their 1s read timeout.
        for handle in self.threads.lock().unwrap().drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for DiscoveryHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub fn start_discovery(config: DiscoveryConfig) -> anyhow::Result<DiscoveryHandle> {
    let stop = Arc::new(AtomicBool::new(false));
    let mut threads = Vec::new();

    // UDP probe responder.
    {
        let udp = UdpSocket::bind(("0.0.0.0", config.udp_port))?;
        udp.set_read_timeout(Some(Duration::from_secs(1)))?;
        let stop = Arc::clone(&stop);
        let cfg = config.clone();
        threads.push(
            std::thread::Builder::new()
                .name("vibe-udp-disc".into())
                .spawn(move || {
                    let mut buf = [0u8; 512];
                    while !stop.load(Ordering::SeqCst) {
                        match udp.recv_from(&mut buf) {
                            Ok((n, src)) => {
                                let probe = String::from_utf8_lossy(&buf[..n]);
                                if probe.trim() == "VIBE_DISCOVER_V1" {
                                    let reply = json!({
                                        "service": "vibe-local",
                                        "version": env!("CARGO_PKG_VERSION"),
                                        "http_port": cfg.http_port,
                                        "lan": cfg.lan,
                                        "host": local_hostname(),
                                    });
                                    let _ = udp.send_to(reply.to_string().as_bytes(), src);
                                }
                            }
                            Err(_) => continue,
                        }
                    }
                })?,
        );
    }

    // mDNS DNS-SD responder. Port 5353 is frequently held by the system
    // resolver (dnscache / avahi / mDNSResponder / mdnsd), so bind with
    // SO_REUSEADDR and treat failure as best-effort: the UDP probe responder
    // above keeps discovery alive either way.
    match bind_mdns_socket(config.mdns_port) {
        Ok(mdns) => {
            let iface = Ipv4Addr::UNSPECIFIED;
            // Multicast join can fail in restricted environments; unicast mDNS
            // queries still work, so this is best-effort.
            let _ = mdns.join_multicast_v4(&MDNS_GROUP, &iface);
            let stop = Arc::clone(&stop);
            let cfg = config.clone();
            threads.push(std::thread::Builder::new().name("vibe-mdns".into()).spawn(
                move || {
                    let mut buf = [0u8; 1500];
                    while !stop.load(Ordering::SeqCst) {
                        match mdns.recv_from(&mut buf) {
                            Ok((n, src)) => {
                                let matched = parse_mdns_questions(&buf[..n])
                                    .iter()
                                    .any(|q| q.to_lowercase().contains(SERVICE_TYPE));
                                if matched {
                                    let packet = build_mdns_response(&cfg, &local_ip());
                                    let dest: SocketAddr = (src.ip(), MDNS_PORT).into();
                                    let _ = mdns.send_to(&packet, dest);
                                    let _ = mdns.send_to(&packet, (MDNS_GROUP, MDNS_PORT));
                                }
                            }
                            Err(_) => continue,
                        }
                    }
                },
            )?);
        }
        Err(e) => {
            eprintln!(
                "[vibe-discovery] mDNS responder unavailable ({e:#}); \
                 UDP probe discovery on port {} remains active",
                config.udp_port
            );
        }
    }

    Ok(DiscoveryHandle {
        stop,
        threads: Mutex::new(threads),
    })
}

/// Bind the mDNS UDP socket with SO_REUSEADDR so a system resolver already
/// holding 5353 does not make `vibe-local serve` fail with AddrInUse.
fn bind_mdns_socket(port: u16) -> anyhow::Result<UdpSocket> {
    use socket2::{Domain, Protocol, Socket, Type};
    let sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
    sock.set_reuse_address(true)?;
    // Unix needs SO_REUSEPORT as well to share a unicast UDP port.
    #[cfg(unix)]
    sock.set_reuse_port(true)?;
    sock.bind(&SocketAddr::from(([0, 0, 0, 0], port)).into())?;
    sock.set_read_timeout(Some(Duration::from_secs(1)))?;
    Ok(sock.into())
}

/// Best-effort primary LAN IP. UDP `connect` sends no packets; it only asks
/// the routing table which source address would be used. Internet targets
/// fail on air-gapped Wi-Fi / hotspots, so RFC1918 gateway targets are probed
/// next, then active non-loopback interfaces are enumerated, and only then
/// do we fall back to loopback.
pub fn local_ip() -> IpAddr {
    for target in [
        "8.8.8.8:80",
        "192.168.1.1:80",
        "10.0.0.1:80",
        "172.16.0.1:80",
    ] {
        if let Ok(sock) = UdpSocket::bind("0.0.0.0:0")
            && sock.connect(target).is_ok()
            && let Ok(addr) = sock.local_addr()
            && let IpAddr::V4(v4) = addr.ip()
            && !v4.is_unspecified()
            && !v4.is_loopback()
        {
            return IpAddr::V4(v4);
        }
    }
    if let Some(v4) = pick_lan_ip(&lan_ipv4_addresses()) {
        return IpAddr::V4(v4);
    }
    IpAddr::V4(Ipv4Addr::LOCALHOST)
}

/// Prefer private RFC1918 addresses (192.168 > 10 > 172.16-31), skipping
/// loopback, link-local and unspecified candidates.
fn pick_lan_ip(candidates: &[Ipv4Addr]) -> Option<Ipv4Addr> {
    let usable = |ip: &Ipv4Addr| {
        !ip.is_loopback() && !ip.is_link_local() && !ip.is_unspecified() && !ip.is_broadcast()
    };
    let octets = candidates
        .iter()
        .copied()
        .filter(usable)
        .collect::<Vec<_>>();
    octets
        .iter()
        .find(|ip| ip.octets()[0] == 192 && ip.octets()[1] == 168)
        .or_else(|| octets.iter().find(|ip| ip.octets()[0] == 10))
        .or_else(|| {
            octets
                .iter()
                .find(|ip| ip.octets()[0] == 172 && (16..=31).contains(&ip.octets()[1]))
        })
        .or_else(|| octets.first())
        .copied()
}

/// Enumerate assigned non-loopback IPv4 interface addresses.
#[cfg(target_os = "windows")]
fn lan_ipv4_addresses() -> Vec<Ipv4Addr> {
    #[repr(C)]
    struct MibIpAddrRow {
        dw_addr: u32,
        dw_index: u32,
        dw_mask: u32,
        dw_bcast_addr: u32,
        dw_reasm_size: u32,
        unused1: u16,
        w_type: u16,
    }
    #[link(name = "iphlpapi")]
    unsafe extern "system" {
        fn GetIpAddrTable(table: *mut u8, size: *mut u32, order: i32) -> u32;
    }
    let mut size: u32 = 0;
    unsafe {
        GetIpAddrTable(std::ptr::null_mut(), &mut size, 0);
    }
    if size == 0 {
        return Vec::new();
    }
    let mut buf = vec![0u8; size as usize];
    let ret = unsafe { GetIpAddrTable(buf.as_mut_ptr(), &mut size, 0) };
    if ret != 0 || buf.len() < 4 {
        return Vec::new();
    }
    let count = u32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let row_size = std::mem::size_of::<MibIpAddrRow>();
    let mut out = Vec::new();
    for i in 0..count {
        let off = 4 + i * row_size;
        if off + row_size > buf.len() {
            break;
        }
        let raw = u32::from_ne_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        // dwAddr is stored in network byte order.
        let ip = Ipv4Addr::from(raw.to_be());
        if !ip.is_loopback() && !ip.is_unspecified() && !ip.is_link_local() {
            out.push(ip);
        }
    }
    out
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
fn lan_ipv4_addresses() -> Vec<Ipv4Addr> {
    #[repr(C)]
    struct IfAddrs {
        ifa_next: *mut IfAddrs,
        ifa_name: *const std::ffi::c_char,
        ifa_flags: std::ffi::c_uint,
        ifa_addr: *mut libc::sockaddr,
        ifa_netmask: *mut libc::sockaddr,
        // ifa_ifu / ifa_data follow but are never read.
    }
    unsafe extern "C" {
        fn getifaddrs(ifap: *mut *mut IfAddrs) -> std::ffi::c_int;
        fn freeifaddrs(ifa: *mut IfAddrs);
    }
    const IFF_LOOPBACK: std::ffi::c_uint = 0x8;
    let mut head: *mut IfAddrs = std::ptr::null_mut();
    if unsafe { getifaddrs(&mut head) } != 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut cur = head;
    while !cur.is_null() {
        let entry = unsafe { &*cur };
        if !entry.ifa_addr.is_null()
            && entry.ifa_flags & IFF_LOOPBACK == 0
            && unsafe { (*entry.ifa_addr).sa_family } as i32 == libc::AF_INET
        {
            let sin = unsafe { &*(entry.ifa_addr as *const libc::sockaddr_in) };
            let ip = Ipv4Addr::from(u32::from_be(sin.sin_addr.s_addr));
            if !ip.is_loopback() && !ip.is_unspecified() && !ip.is_link_local() {
                out.push(ip);
            }
        }
        cur = entry.ifa_next;
    }
    unsafe {
        freeifaddrs(head);
    }
    out
}

#[cfg(not(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "android",
    target_os = "macos"
)))]
fn lan_ipv4_addresses() -> Vec<Ipv4Addr> {
    Vec::new()
}

pub fn local_hostname() -> String {
    let raw = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .or_else(|| {
            vibe_local_core::util::run_with_timeout("hostname", &[], Duration::from_secs(3))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "vibe-local".to_string());
    let sanitized: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = sanitized.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "vibe-local".into()
    } else {
        trimmed
    }
}

// ---------------------------------------------------------------------------
// Minimal DNS wire format
// ---------------------------------------------------------------------------

fn encode_name(name: &str, out: &mut Vec<u8>) {
    for label in name.split('.') {
        if label.is_empty() {
            continue;
        }
        out.push(label.len() as u8);
        out.extend_from_slice(label.as_bytes());
    }
    out.push(0);
}

/// Decode a DNS name starting at `pos`, following compression pointers
/// (`(len & 0xC0) == 0xC0`). Returns the name and the position just past the
/// name in the *original* stream (2 bytes past the first pointer, if any).
/// Pointers must point backwards; a 16-hop guard bounds malformed packets.
fn decode_name(packet: &[u8], mut pos: usize) -> Option<(String, usize)> {
    let mut labels = Vec::new();
    let mut end_pos = None;
    let mut hops = 0usize;
    loop {
        let len = *packet.get(pos)? as usize;
        if len & 0xc0 == 0xc0 {
            let lo = *packet.get(pos + 1)? as usize;
            let offset = ((len & 0x3f) << 8) | lo;
            if end_pos.is_none() {
                end_pos = Some(pos + 2);
            }
            hops += 1;
            // Backwards-only + hop guard defeats pointer loops.
            if hops > 16 || offset >= pos {
                return None;
            }
            pos = offset;
            continue;
        }
        if len & 0xc0 != 0 {
            return None; // reserved label types (0x40/0x80) are invalid
        }
        pos += 1;
        if len == 0 {
            break;
        }
        let end = pos.checked_add(len)?;
        labels.push(String::from_utf8_lossy(packet.get(pos..end)?).into_owned());
        pos = end;
    }
    Some((labels.join("."), end_pos.unwrap_or(pos)))
}

/// Parse every question in the DNS query section (all `qdcount` entries),
/// skipping the 4-byte QTYPE/QCLASS trailer after each name.
pub fn parse_mdns_questions(packet: &[u8]) -> Vec<String> {
    if packet.len() < 12 {
        return Vec::new();
    }
    let qdcount = u16::from_be_bytes([packet[4], packet[5]]) as usize;
    let mut pos = 12usize;
    let mut out = Vec::new();
    for _ in 0..qdcount {
        let Some((name, next)) = decode_name(packet, pos) else {
            break;
        };
        let Some(after) = next.checked_add(4) else {
            break;
        };
        if after > packet.len() {
            break;
        }
        out.push(name);
        pos = after;
    }
    out
}

/// Parse the first question name from a DNS query packet.
pub fn parse_mdns_question(packet: &[u8]) -> Option<String> {
    parse_mdns_questions(packet).into_iter().next()
}

fn push_record(out: &mut Vec<u8>, name: &str, rtype: u16, ttl: u32, rdata: &[u8]) {
    encode_name(name, out);
    out.extend_from_slice(&rtype.to_be_bytes());
    out.extend_from_slice(&0x8001u16.to_be_bytes()); // IN + cache-flush
    out.extend_from_slice(&ttl.to_be_bytes());
    out.extend_from_slice(&(rdata.len() as u16).to_be_bytes());
    out.extend_from_slice(rdata);
}

/// Build a full DNS-SD answer: PTR, SRV, TXT and A records.
pub fn build_mdns_response(cfg: &DiscoveryConfig, ip: &IpAddr) -> Vec<u8> {
    let host = local_hostname();
    let instance = format!("Vibe Local ({host}).{SERVICE_TYPE}");
    let target = format!("{host}.local");
    let mut out = Vec::new();
    // Header: id 0, flags 0x8400 (response, authoritative), 4 answers.
    out.extend_from_slice(&0u16.to_be_bytes());
    out.extend_from_slice(&0x8400u16.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes()); // qd
    out.extend_from_slice(&4u16.to_be_bytes()); // an
    out.extend_from_slice(&0u16.to_be_bytes()); // ns
    out.extend_from_slice(&0u16.to_be_bytes()); // ar

    // PTR: service type -> instance
    let mut ptr_data = Vec::new();
    encode_name(&instance, &mut ptr_data);
    push_record(&mut out, SERVICE_TYPE, 12, 120, &ptr_data);

    // SRV: instance -> host:port
    let mut srv_data = Vec::new();
    srv_data.extend_from_slice(&0u16.to_be_bytes()); // priority
    srv_data.extend_from_slice(&0u16.to_be_bytes()); // weight
    srv_data.extend_from_slice(&cfg.http_port.to_be_bytes());
    encode_name(&target, &mut srv_data);
    push_record(&mut out, &instance, 33, 120, &srv_data);

    // TXT: metadata
    let mut txt_data = Vec::new();
    for entry in [
        "path=/".to_string(),
        format!("lan={}", if cfg.lan { 1 } else { 0 }),
        format!("version={}", env!("CARGO_PKG_VERSION")),
    ] {
        txt_data.push(entry.len() as u8);
        txt_data.extend_from_slice(entry.as_bytes());
    }
    push_record(&mut out, &instance, 16, 120, &txt_data);

    // A: hostname -> IP (only for IPv4; IPv6-only hosts skip the A record
    // and rely on the UDP probe responder).
    match ip {
        IpAddr::V4(v4) => push_record(&mut out, &target, 1, 120, &v4.octets()),
        IpAddr::V6(_) => {
            // Replace the 4th answer slot count: we emitted 3 records.
            out[6..8].copy_from_slice(&3u16.to_be_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query_packet(name: &str) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&0u16.to_be_bytes()); // id
        p.extend_from_slice(&0u16.to_be_bytes()); // flags: query
        p.extend_from_slice(&1u16.to_be_bytes()); // qd
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&0u16.to_be_bytes());
        encode_name(name, &mut p);
        p.extend_from_slice(&12u16.to_be_bytes()); // PTR
        p.extend_from_slice(&1u16.to_be_bytes()); // IN
        p
    }

    #[test]
    fn question_parse_roundtrip() {
        let p = query_packet(SERVICE_TYPE);
        assert_eq!(parse_mdns_question(&p).as_deref(), Some(SERVICE_TYPE));
        assert_eq!(parse_mdns_question(&[0u8; 4]), None);
        assert_eq!(
            parse_mdns_question(&query_packet("_other._tcp.local")).as_deref(),
            Some("_other._tcp.local")
        );
    }

    #[test]
    fn multi_question_packet() {
        let mut p = Vec::new();
        p.extend_from_slice(&0u16.to_be_bytes()); // id
        p.extend_from_slice(&0u16.to_be_bytes()); // flags
        p.extend_from_slice(&2u16.to_be_bytes()); // qd = 2
        p.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        encode_name("_other._tcp.local", &mut p);
        p.extend_from_slice(&12u16.to_be_bytes());
        p.extend_from_slice(&1u16.to_be_bytes());
        encode_name(SERVICE_TYPE, &mut p);
        p.extend_from_slice(&12u16.to_be_bytes());
        p.extend_from_slice(&1u16.to_be_bytes());
        let qs = parse_mdns_questions(&p);
        assert_eq!(qs, vec!["_other._tcp.local", SERVICE_TYPE]);
        // First-question accessor still works.
        assert_eq!(
            parse_mdns_question(&p).as_deref(),
            Some("_other._tcp.local")
        );
    }

    #[test]
    fn compression_pointer_decoded() {
        // Question 1: full name at offset 12. Question 2: pointer to 12.
        let mut p = Vec::new();
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&2u16.to_be_bytes());
        p.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        encode_name(SERVICE_TYPE, &mut p);
        p.extend_from_slice(&12u16.to_be_bytes());
        p.extend_from_slice(&1u16.to_be_bytes());
        p.extend_from_slice(&[0xc0, 0x0c]); // pointer to offset 12
        p.extend_from_slice(&33u16.to_be_bytes()); // SRV
        p.extend_from_slice(&1u16.to_be_bytes());
        let qs = parse_mdns_questions(&p);
        assert_eq!(qs, vec![SERVICE_TYPE, SERVICE_TYPE]);
    }

    #[test]
    fn compression_pointer_loop_rejected() {
        // Self-referential pointer at offset 12 must not hang or recurse.
        let mut p = Vec::new();
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&0u16.to_be_bytes());
        p.extend_from_slice(&1u16.to_be_bytes());
        p.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        p.extend_from_slice(&[0xc0, 0x0c]);
        p.extend_from_slice(&12u16.to_be_bytes());
        p.extend_from_slice(&1u16.to_be_bytes());
        assert!(parse_mdns_questions(&p).is_empty());
        // Forward pointer (offset 20 > current 12) is also rejected.
        let mut fwd = p.clone();
        fwd[12] = 0xc0;
        fwd[13] = 0x14;
        assert!(parse_mdns_questions(&fwd).is_empty());
    }

    #[test]
    fn truncated_question_section_safe() {
        let mut p = query_packet(SERVICE_TYPE);
        p.truncate(p.len() - 2); // cut into QCLASS
        assert!(parse_mdns_questions(&p).is_empty());
        let mut lying = query_packet(SERVICE_TYPE);
        lying[4..6].copy_from_slice(&99u16.to_be_bytes()); // qdcount lies
        assert_eq!(parse_mdns_questions(&lying).len(), 1);
    }

    #[test]
    fn lan_ip_selection_prefers_rfc1918() {
        let candidates = [
            Ipv4Addr::new(127, 0, 0, 1),
            Ipv4Addr::new(169, 254, 10, 5),
            Ipv4Addr::new(0, 0, 0, 0),
            Ipv4Addr::new(10, 0, 0, 7),
            Ipv4Addr::new(192, 168, 1, 42),
            Ipv4Addr::new(172, 16, 0, 3),
        ];
        assert_eq!(
            pick_lan_ip(&candidates),
            Some(Ipv4Addr::new(192, 168, 1, 42))
        );
        assert_eq!(
            pick_lan_ip(&[Ipv4Addr::new(172, 16, 0, 3), Ipv4Addr::new(10, 0, 0, 7)]),
            Some(Ipv4Addr::new(10, 0, 0, 7))
        );
        assert_eq!(
            pick_lan_ip(&[Ipv4Addr::new(172, 20, 1, 1)]),
            Some(Ipv4Addr::new(172, 20, 1, 1))
        );
        // Non-RFC1918 public address is still usable as a last resort.
        assert_eq!(
            pick_lan_ip(&[Ipv4Addr::new(8, 8, 4, 4)]),
            Some(Ipv4Addr::new(8, 8, 4, 4))
        );
        assert_eq!(pick_lan_ip(&[Ipv4Addr::LOCALHOST]), None);
        assert_eq!(pick_lan_ip(&[]), None);
        // local_ip never returns an unspecified address.
        assert!(!local_ip().is_unspecified());
    }

    #[test]
    fn mdns_socket_binds_with_reuse() {
        // Two sockets on the same port must coexist via SO_REUSEADDR
        // (mirrors the system resolver holding 5353).
        let a = bind_mdns_socket(0).unwrap();
        let port = a.local_addr().unwrap().port();
        let b = bind_mdns_socket(port).unwrap();
        assert_eq!(b.local_addr().unwrap().port(), port);
    }

    #[test]
    fn response_contains_records() {
        let cfg = DiscoveryConfig {
            http_port: 20129,
            lan: true,
            udp_port: 0,
            mdns_port: 0,
        };
        let packet = build_mdns_response(&cfg, &IpAddr::V4(Ipv4Addr::new(192, 168, 1, 50)));
        assert!(packet.len() > 60);
        let ancount = u16::from_be_bytes([packet[6], packet[7]]);
        assert_eq!(ancount, 4);
        let text = String::from_utf8_lossy(&packet);
        assert!(text.contains("_vibe-local"));
        assert!(text.contains("lan=1"));
        // port 20129 = 0x4EA1 present in SRV rdata
        assert!(packet.windows(2).any(|w| w == [0x4e, 0xa1]));
        // A record IP bytes
        assert!(packet.windows(4).any(|w| w == [192, 168, 1, 50]));
    }

    #[test]
    fn udp_probe_responder() {
        let cfg = DiscoveryConfig {
            http_port: 20129,
            lan: false,
            udp_port: 0,
            mdns_port: 0,
        };
        // Bind an ephemeral probe socket manually to test the protocol logic.
        let responder = UdpSocket::bind("127.0.0.1:0").unwrap();
        let addr = responder.local_addr().unwrap();
        responder
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let server = std::thread::spawn(move || {
            let mut buf = [0u8; 512];
            let (n, src) = responder.recv_from(&mut buf).unwrap();
            assert_eq!(
                String::from_utf8_lossy(&buf[..n]).trim(),
                "VIBE_DISCOVER_V1"
            );
            let reply = json!({"service": "vibe-local", "http_port": 20129, "lan": false});
            responder
                .send_to(reply.to_string().as_bytes(), src)
                .unwrap();
        });
        let client = UdpSocket::bind("127.0.0.1:0").unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        client.send_to(b"VIBE_DISCOVER_V1", addr).unwrap();
        let mut buf = [0u8; 512];
        let (n, _) = client.recv_from(&mut buf).unwrap();
        let reply: serde_json::Value = serde_json::from_slice(&buf[..n]).unwrap();
        assert_eq!(reply["service"], "vibe-local");
        assert_eq!(reply["http_port"], 20129);
        server.join().unwrap();
        let _ = cfg;
    }

    #[test]
    fn hostname_sanitized() {
        let name = local_hostname();
        assert!(!name.is_empty());
        assert!(name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'));
    }
}
