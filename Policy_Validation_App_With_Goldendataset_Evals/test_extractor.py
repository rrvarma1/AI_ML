import pytest
from pydantic import ValidationError

from extractor import (
    DealTerms,
    extract_deal_terms,
    extract_deal_terms_model,
    extraction_quality,
    run_extractor,
    run_extractor_fallback,
)
from inspect import signature
import json


def test_extracts_and_calculates_terms():
    text = """Borrower Name: ABC Manufacturing Ltd.
Deal Type: Business Loan
Loan Amount: INR 5,000,000
Interest Rate: 11.5%
Term: 5 years
Collateral Value: INR 7,000,000
DSCR: 1.32x
Purpose: Plant expansion
Jurisdiction: India"""
    terms = extract_deal_terms(text)
    assert terms["borrower_name"] == "ABC Manufacturing Ltd"
    assert terms["currency"] == "INR"
    assert terms["term_years"] == 5
    assert terms["term_months"] == 60
    assert terms["stated_ltv"] is None
    assert extraction_quality(terms)["status"] == "Extraction completed"


def test_returns_pydantic_contract():
    result = extract_deal_terms_model("Borrower: Example Ltd\nLoan Amount: INR 1000\nCollateral Value: INR 2000")
    assert isinstance(result, DealTerms)
    assert result.stated_ltv is None


def test_does_not_make_business_validation_decisions():
    result = DealTerms(loan_amount=1_000_000, collateral_realisable_value=2_000_000, stated_ltv=80)
    assert result.stated_ltv == 80


def test_fallback_uses_same_validated_contract():
    output = run_extractor_fallback("Borrower: Example Ltd\nLoan Amount: INR 1000\nCollateral Value: INR 2000")
    assert isinstance(output.terms, DealTerms)
    assert output.extraction_method == "deterministic_fallback"


def test_requested_financial_fields_from_sample():
    text = open("sample_deal.txt", encoding="utf-8").read()
    terms = extract_deal_terms(text)
    assert terms["loan_amount"] == 50_000_000
    assert terms["term_years"] == 5
    assert terms["term_months"] == 60
    assert terms["interest_rate_pct"] == 10.25
    assert terms["interest_amount"] is None
    assert terms["estimated_dscr"] == 1.42
    assert terms["total_debt"] == 63_000_000
    assert terms["tangible_net_worth"] == 35_000_000
    assert terms["collateral_market_value"] == 72_000_000
    assert terms["collateral_realisable_value"] == 57_500_000
    assert terms["stated_ltv"] is None
    assert terms["stated_collateral_coverage"] == 1.15
    assert terms["monthly_principal_instalment"] == 925_926
    assert terms["number_of_instalments"] == 54
    assert terms["total_facility_amount"] == 50_000_000


@pytest.mark.parametrize("label", [
    "Debt-to-Equity Ratio",
    "Debt to Equity Ratio",
    "Debt/Equity Ratio",
    "Debt-Equity Ratio",
    "Debt Equity Ratio",
    "D/E Ratio",
])
def test_extracts_debt_to_equity_label_variants(label):
    terms = extract_deal_terms(f"{label}: 1.80x")
    assert terms["debt_to_equity"] == 1.80


def test_default_llm_is_local_gemma():
    from extractor import run_extractor
    defaults = signature(run_extractor).parameters
    assert defaults["provider"].default == "ollama"
    assert defaults["model_name"].default == "gemma3:4b"


def test_ollama_json_mode_avoids_grammar_sampler(monkeypatch):
    from extractor import DealTerms, run_extractor

    payload = DealTerms(borrower_name="Example Ltd", loan_amount=1_000_000, currency="INR").model_dump_json()

    class Response:
        content = payload

    class FakeOllama:
        def invoke(self, messages):
            return Response()

        def with_structured_output(self, *args, **kwargs):
            raise AssertionError("Ollama must not use grammar-constrained structured output")

    monkeypatch.setattr("extractor._llm", lambda provider, model_name: FakeOllama())
    result = run_extractor("Borrower Name: Example Ltd", provider="ollama", model_name="gemma3:4b")
    assert result.terms.loan_amount == 1_000_000
    assert result.model_name == "ollama:gemma3:4b"


def test_ollama_retries_an_empty_semantic_result(monkeypatch):
    from extractor import run_extractor

    responses = iter([
        "{}",
        '{"borrower_name":"GreenLeaf Packaging Private Limited","loan_amount":50000000,"currency":"INR"}',
    ])

    class Response:
        def __init__(self, content):
            self.content = content

    class FakeOllama:
        def invoke(self, messages):
            return Response(next(responses))

    monkeypatch.setattr("extractor._llm", lambda provider, model_name: FakeOllama())
    result = run_extractor("Borrower Name: GreenLeaf Packaging Private Limited\nLoan Amount: INR 5,00,00,000")
    assert result.terms.loan_amount == 50_000_000
    assert any("targeted retry" in warning for warning in result.warnings)


def test_label_recovery_fills_null_llm_fields(monkeypatch):
    from extractor import run_extractor

    class Response:
        content = "{}"

    class EmptyOllama:
        def invoke(self, messages):
            return Response()

    monkeypatch.setattr("extractor._llm", lambda provider, model_name: EmptyOllama())
    result = run_extractor("Borrower Name: GreenLeaf Packaging Private Limited\nLoan Amount: INR 5,00,00,000\nInterest Rate: 10.25%\nLoan Term: 5 years\nPurpose: Machinery purchase")
    assert result.terms.borrower_name == "GreenLeaf Packaging Private Limited"
    assert result.terms.loan_amount == 50_000_000
    assert result.terms.interest_rate_pct == 10.25
    assert result.terms.term_months == 60
    assert any("recovered" in warning for warning in result.warnings)


def test_explicit_labels_correct_cross_field_llm_substitution(monkeypatch):
    """PVC-GD-027: Total Debt must not replace Loan Amount or its currency."""
    wrong_llm_payload = DealTerms(
        borrower_name="GreenLeaf Packaging Private Limited",
        currency="INR",
        loan_amount=63_000_000,
        total_debt=63_000_000,
        total_facility_amount=63_000_000,
    ).model_dump_json()

    class Response:
        content = wrong_llm_payload

    class WrongOllama:
        def invoke(self, messages):
            return Response()

    monkeypatch.setattr("extractor._llm", lambda provider, model_name: WrongOllama())
    text = """Borrower Name: GreenLeaf Packaging Private Limited
Loan Amount: USD 50000000
Total Debt: INR 63000000
Total Facility Amount: INR 50000000"""
    result = run_extractor(text)

    assert result.terms.currency == "USD"
    assert result.terms.loan_amount == 50_000_000
    assert result.terms.total_debt == 63_000_000
    assert result.terms.total_facility_amount == 50_000_000
    assert any("corrected" in warning for warning in result.warnings)


def test_term_years_and_months_have_independent_label_binding():
    terms = extract_deal_terms("Term Years: 5\nTerm Months: 60")
    assert terms["term_years"] == 5
    assert terms["term_months"] == 60


def test_all_null_extraction_is_not_reported_as_success(monkeypatch):
    from extractor import run_extractor

    class Response:
        content = "{}"

    class EmptyOllama:
        def invoke(self, messages):
            return Response()

    monkeypatch.setattr("extractor._llm", lambda provider, model_name: EmptyOllama())
    with pytest.raises(ValueError, match="extractor returned no values"):
        run_extractor("This non-empty text contains no recognisable deal terms.")
