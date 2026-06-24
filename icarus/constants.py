"""Shared constants. Leaf module — imports nothing from the package, so everything
else can import the field vocabulary from here without circular imports."""

HOLD = 3                                              # frames an order survives unsighted
FIELDS = ("type", "straat", "postcode", "huisnummer")  # chained fields; ordernr is the key
TYPES = ("Delivery", "Carryout", "Dinein")
NO_ADDRESS_TYPES = ("Carryout", "Dinein")             # no delivery address -> excluded from address OCR

DEFAULT_THRESHOLDS = {
    "ordernr": 5, "type": 2, "straat": 3, "postcode": 3, "huisnummer": 3,
    "type_cutoff": 0.5, "straat_cutoff": 0.85, "postcode_cutoff": 0.5, "huisnummer_cutoff": 1.0,
}
