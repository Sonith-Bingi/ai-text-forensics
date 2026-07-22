from forensics.data.schema import parse_src


def test_parse_human_row():
    assert parse_src("yelp_human") == ("yelp", "human", "human")


def test_parse_machine_continuation():
    assert parse_src("yelp_machine_continuation_opt_13b") == ("yelp", "opt_13b", "continuation")


def test_parse_machine_specified():
    assert parse_src("cmv_machine_specified_gpt-3.5-trubo") == ("cmv", "gpt-3.5-trubo", "specified")


def test_parse_machine_topical():
    assert parse_src("wp_machine_topical_text-davinci-003") == ("wp", "text-davinci-003", "topical")


def test_parse_gpt4_direct():
    assert parse_src("cnn_gpt4") == ("cnn", "gpt4", "direct")


def test_parse_gpt4_human_paraphrase():
    assert parse_src("cnn_human_para") == ("cnn", "gpt4", "paraphrase_of_human")


def test_parse_gpt4_machine_paraphrase():
    assert parse_src("pubmed_gpt4_para") == ("pubmed", "gpt4", "paraphrase_of_machine")


def test_parse_unknown_domain():
    assert parse_src("totally_unknown_src")[0] == "unknown"


def test_domain_prefix_disambiguation_sci_gen():
    # sci_gen must not be mis-parsed by a shorter domain prefix accidentally matching
    assert parse_src("sci_gen_human")[0] == "sci_gen"
