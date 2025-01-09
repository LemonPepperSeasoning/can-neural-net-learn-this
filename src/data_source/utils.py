from hashlib import sha256


def sha256(data):
    return sha256(data).hexdigest()
    pass
