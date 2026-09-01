from nlp_evaluation import rank_errors


def test_errors_are_ranked_by_confidence():
    errors = rank_errors(
        ["a", "b", "c", "d"],
        [0, 1, 1, 0],
        [0.99, 0.01, 0.7, 0.2],
    )
    assert [error.index for error in errors] == [0, 1]
    assert errors[0].kind == "false_positive"
    assert errors[1].kind == "false_negative"

