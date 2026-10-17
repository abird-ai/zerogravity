# Synthetic acquisition TLS fixtures

Generated locally with OpenSSL for loopback tests only. ca.pem is a private test
trust root; server.pem covers127.0.0.1 and server.key is a public, disposable test
key, not a credential. Tests must validate the chain and hostname. No production
client trusts this CA. The CA signing key was temporary and is not retained.
