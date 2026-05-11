from __future__ import annotations

from ..cli import _dashboard_urls, _is_probably_routable_lan_ip, detect_lan_addresses


def test_dashboard_urls_bind_all_includes_loopback_and_each_lan_ip() -> None:
	urls = _dashboard_urls(
		bind_host="0.0.0.0", port=8050, lan_addresses=["10.0.0.5", "192.168.1.20"]
	)
	assert urls == [
		"http://127.0.0.1:8050/",
		"http://10.0.0.5:8050/",
		"http://192.168.1.20:8050/",
	]


def test_dashboard_urls_bind_loopback_only_returns_one_url() -> None:
	assert _dashboard_urls(bind_host="127.0.0.1", port=8050, lan_addresses=["10.0.0.5"]) == [
		"http://127.0.0.1:8050/"
	]
	assert _dashboard_urls(bind_host="localhost", port=8050, lan_addresses=[]) == [
		"http://localhost:8050/"
	]


def test_dashboard_urls_explicit_host_returns_one_url() -> None:
	# An explicit LAN IP is "what the user asked for" — don't echo a stale 127.0.0.1.
	urls = _dashboard_urls(bind_host="10.0.0.5", port=9001, lan_addresses=["1.2.3.4"])
	assert urls == ["http://10.0.0.5:9001/"]


def test_routable_lan_ip_filter_drops_loopback_link_local_docker_bridge() -> None:
	assert not _is_probably_routable_lan_ip("")
	assert not _is_probably_routable_lan_ip("127.0.0.1")
	assert not _is_probably_routable_lan_ip("127.1.2.3")
	assert not _is_probably_routable_lan_ip("169.254.1.2")  # RFC 3927 link-local
	assert not _is_probably_routable_lan_ip("172.17.0.1")  # default Docker bridge
	assert not _is_probably_routable_lan_ip("::1")  # IPv6 skipped (contains ':')
	# Real LAN ranges pass through.
	assert _is_probably_routable_lan_ip("10.0.0.5")
	assert _is_probably_routable_lan_ip("192.168.1.20")
	assert _is_probably_routable_lan_ip("172.20.10.5")  # real 172.16/12 LAN
	# Explicit public addresses also pass — caller can still serve there.
	assert _is_probably_routable_lan_ip("203.0.113.10")


def test_detect_lan_addresses_returns_list_of_strings() -> None:
	addrs = detect_lan_addresses()
	assert isinstance(addrs, list)
	for ip in addrs:
		assert isinstance(ip, str)
		# Whatever shows up must already have survived the filter.
		assert _is_probably_routable_lan_ip(ip)
