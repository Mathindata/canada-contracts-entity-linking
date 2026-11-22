from cel.normalize import clean_name, clean_postal, initials


def test_legal_suffixes_en_fr():
    assert clean_name("Summit Data Solutions Inc.") == "SUMMIT DATA SOLUTIONS"
    assert clean_name("SUMMIT DATA SOLUTIONS, LTD.") == "SUMMIT DATA SOLUTIONS"
    assert clean_name("Services Boréal Ltée") == "SERVICES BOREAL"
    assert clean_name("Smith & Jones S.E.N.C.R.L.") == "SMITH AND JONES"


def test_the_ampersand_and_letters():
    assert clean_name("The Cedar & Maple Group") == "CEDAR AND MAPLE GROUP"
    assert clean_name("I.B.M. Canada Ltd") == "IBM CANADA"


def test_suffix_alone_is_kept():
    assert clean_name("Inc.") == "INC"


def test_initials():
    assert initials("NORTHERN DATA SYSTEMS") == "NDS"
    assert initials("BUREAU DE LA TRADUCTION") == "BT"
    assert initials("IBM") == ""


def test_postal():
    assert clean_postal("k1a 0b1") == "K1A"
    assert clean_postal("NA") == ""
    assert clean_postal("") == ""
    assert clean_postal("12345") == ""
