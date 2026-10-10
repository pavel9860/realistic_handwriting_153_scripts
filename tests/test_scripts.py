from hwsynth import scripts as S


def test_script_lookup():
    assert S.script("a") == "Latin"
    assert S.script("я") == "Cyrillic"
    assert S.script("你") == "Han"
    assert S.script("ש") == "Hebrew"
    assert S.script("ب") == "Arabic"
    assert S.script("5") == "Common"
    assert S.script("́") == "Inherited"


def test_text_script_and_direction():
    assert S.text_script("Привет, world!") == "Cyrillic"
    assert S.text_script("123 !?") == "Common"
    assert S.text_rtl("שלום עולם")
    assert not S.text_rtl("hello שלום world")


def test_units_cluster_or_word():
    assert S.units("ёж", "Cyrillic") == ["ё", "ж"]
    assert S.units("ét", "Latin") == ["é", "t"]
    assert S.units("नमस्ते", "Devanagari") == ["नमस्ते"]


def test_visual_order_keeps_digits():
    assert S.visual_units("אב12", "Hebrew") == ["1", "2", "ב", "א"]
    assert S.visual_units("abc", "Latin") == ["a", "b", "c"]


def test_scripts_table():
    names = S.all_scripts()
    assert len(names) > 150 and "Latin" in names and "Common" not in names
    assert S.letters("Greek", 5)
