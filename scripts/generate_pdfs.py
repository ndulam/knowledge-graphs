"""Generate sample compliance PDF documents for unstructured RAG ingestion."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"

# ── Document content ──────────────────────────────────────────────────────────

_AML_POLICY = {
    "title": "Anti-Money Laundering (AML) Policy",
    "subtitle": "Financial Compliance Division — Policy Reference FCD-AML-001 | Version 3.2",
    "sections": [
        (
            "1. Purpose and Scope",
            """
This Anti-Money Laundering Policy establishes the framework for detecting, preventing, and
reporting money laundering activities within our financial institution. This policy applies
to all business units, advisors, and customer-facing staff.

Money laundering is the process by which criminals disguise the proceeds of crime to make them
appear legitimate. It typically occurs in three stages: placement (introducing illicit funds
into the financial system), layering (disguising the trail through complex transactions), and
integration (reintroducing the funds into the legitimate economy).
""",
        ),
        (
            "2. Customer Risk Classification",
            """
All customers must be assigned a risk score between 0.0 and 1.0 at onboarding and reviewed
quarterly. The risk score reflects the likelihood that the customer's activity involves money
laundering or financial crime.

Risk tiers:
- Low Risk (0.0 to 0.49): Standard due diligence. Annual review. No enhanced monitoring required.
- Medium Risk (0.50 to 0.79): Periodic review every six months. Transaction monitoring alerts enabled.
- High Risk (0.80 to 1.00): Enhanced Due Diligence (EDD) required immediately. Transactions
  exceeding $5,000 require manual review. Quarterly face-to-face review with assigned advisor.
  Immediate escalation to compliance if risk score exceeds 0.90.

Customers with a risk score at or above 0.80 are classified as High Risk Customers (HRC) and
are subject to Enhanced Due Diligence (EDD) requirements as defined in Section 4.
""",
        ),
        (
            "3. Transaction Monitoring Rules",
            """
The following transaction monitoring rules apply to all accounts:

3.1 Currency Transaction Reporting (CTR): All cash transactions exceeding $10,000 in a single
business day must be reported to FinCEN within 15 calendar days of the transaction date.

3.2 Suspicious Activity Reports (SARs): Staff must file a SAR within 30 days of detecting
suspicious activity. SARs are required when:
- A transaction involves $5,000 or more and there is reason to suspect involvement in illegal
  activity.
- The customer is structuring transactions to avoid reporting thresholds.
- Transactions are inconsistent with the customer's known business profile.

3.3 Structuring (Smurfing): It is illegal for customers to deliberately structure transactions
to avoid the $10,000 CTR reporting threshold. Red flags include:
- Multiple cash deposits on the same day, each below $10,000.
- A large inflow exceeding $10,000 followed by multiple smaller outgoing transactions with an
  average below $5,000 constitutes a prima facie structuring pattern requiring immediate SAR
  review.
- Multiple accounts receiving coordinated transfers from a single source.
- Frequent round-number transactions or transactions that appear designed to split a larger amount.

3.4 Wire Transfer Monitoring: Incoming wire transfers exceeding $3,000 must include complete
originator information. Cross-border transfers to or from high-risk jurisdictions require
additional documentation.
""",
        ),
        (
            "4. Enhanced Due Diligence (EDD)",
            """
Enhanced Due Diligence is mandatory for all customers classified as High Risk (risk score
at or above 0.80). EDD requirements include:

4.1 Source of Funds Verification: Documentary evidence of the source of all funds deposited
exceeding $5,000.

4.2 Beneficial Ownership: Full beneficial ownership disclosure for accounts with beneficial
owners holding more than 10% interest.

4.3 Ongoing Transaction Review: All outgoing transactions must be reviewed within 48 hours.
Suspicious patterns must be escalated to the BSA Officer immediately.

4.4 Quarterly Review Meetings: The assigned financial advisor must conduct a quarterly review
with each HRC client and document findings in the compliance system.

4.5 Relationship Termination: If EDD requirements cannot be satisfied within 90 days of the
HRC designation, the institution must consider terminating the business relationship and
filing a SAR.
""",
        ),
        (
            "5. Geographic Risk Considerations",
            """
Transaction risk is elevated for transfers involving the following country risk tiers:

High-Risk Jurisdictions: Transactions involving jurisdictions on the FATF blacklist or grey
list require automatic SAR consideration regardless of amount.

Elevated-Risk Countries in our customer base carry enhanced transaction monitoring requirements:
- Brazil (BR): Elevated due to cross-border structuring patterns observed historically.
- Singapore (SG): Monitor closely for trade-based money laundering schemes.
- India (IN): Enhanced documentation requirements for wire transfers.
- France (FR): Standard EU AML Directive compliance applies.

Standard-Risk Countries: United States (US), United Kingdom (UK), Australia (AU), Germany (DE),
Canada (CA), and Japan (JP) operate under standard AML monitoring protocols consistent with
their domestic regulatory frameworks.
""",
        ),
        (
            "6. Reporting and Escalation",
            """
6.1 Internal Escalation: Any employee who identifies a suspicious transaction pattern must
immediately notify their compliance officer. Failure to report is subject to disciplinary
action and personal regulatory liability.

6.2 SAR Filing: SARs must be filed electronically through FinCEN's BSA E-Filing System.
The minimum threshold for SAR filing is $5,000 for known or suspected violations, and
$25,000 when the subject cannot be identified.

6.3 Record Retention: All transaction records, EDD documentation, and SAR filings must be
retained for a minimum of five years from the date of the last transaction.

6.4 Confidentiality: The existence of a SAR filing must not be disclosed to the subject of
the report or any third party. Tipping off a suspect is a criminal offense.
""",
        ),
    ],
}

_RISK_FRAMEWORK = {
    "title": "Customer Risk Assessment Framework",
    "subtitle": "Risk Management Division — Reference RMD-CRA-002 | Version 2.1",
    "sections": [
        (
            "1. Risk Score Methodology",
            """
The Customer Risk Score is a composite index between 0.0 and 1.0 that quantifies the
probability of a customer engaging in financial crime. Scores are computed by the Risk
Analytics Engine using the following weighted factors:

Transaction Behavior (40%): Frequency, size distribution, counterparty diversity, and velocity
of transactions. Unusual patterns such as round-number transactions, structured amounts, or
sudden spikes in activity increase the score.

Customer Profile (30%): Country of residence, occupation, account age, PEP (Politically
Exposed Person) status, and source of wealth declarations.

Network Exposure (20%): Connections to known high-risk entities, counterparties with elevated
risk scores, or involvement in flagged transaction chains.

Account Type (10%): Brokerage and investment accounts with high turnover carry a higher base
risk than standard savings or checking accounts.
""",
        ),
        (
            "2. High-Risk Customer Designation",
            """
A customer is automatically designated High Risk Customer (HRC) when their composite risk
score reaches or exceeds 0.80. Upon HRC designation:

- Immediate notification is sent to the customer's assigned financial advisor.
- The account is flagged in the transaction monitoring system for enhanced surveillance.
- All outgoing transactions over $5,000 enter a 48-hour review queue.
- The compliance team receives an HRC alert within one business day.

Customers with a risk score between 0.90 and 1.00 are classified as Critical Risk and require
escalation to the Chief Compliance Officer within 24 hours of designation. All transactions
from Critical Risk accounts are subject to real-time monitoring.
""",
        ),
        (
            "3. Advisor Portfolio Risk Concentration",
            """
Financial advisors are responsible for monitoring the aggregate risk profile of their managed
client portfolios. Portfolio risk concentration rules:

3.1 Maximum HRC Concentration: No single advisor's portfolio should exceed 50% High Risk
Customers. An advisor whose portfolio reaches 50% HRC triggers an automatic portfolio review.

3.2 100% HRC Portfolio: An advisor managing a portfolio where all clients are classified as
High Risk is in violation of portfolio diversification requirements and must be reviewed by
the Head of Compliance within five business days. This scenario is treated as a potential
systemic risk indicator and may indicate deliberate accumulation of high-risk clients.

3.3 Annual Certification: Advisors must certify annually that they have reviewed all client
risk scores, completed required EDD for HRC clients, and reported any suspicious activity.

3.4 Advisor Indirect Risk Exposure Score: Each advisor is assigned an indirect risk exposure
score calculated as: (number of HRC clients divided by total clients) multiplied by the
average HRC risk score. Advisors with an indirect exposure score above 0.70 are subject to
enhanced supervision and mandatory quarterly reporting to the compliance team.
""",
        ),
        (
            "4. Network Risk Analysis",
            """
Beyond individual customer scores, the risk framework evaluates network-level risk patterns:

4.1 Risk Contagion: A low-risk customer (score below 0.30) who directly transacts with a
high-risk customer (score above 0.80) must have their risk score reviewed within 30 days.
The transaction creates a contagion flag in the system. If the low-risk customer conducts
three or more transactions with high-risk counterparties within a 90-day window, their risk
score is automatically escalated.

4.2 Circular Transaction Patterns: Three or more accounts forming a closed transaction ring
(Account A sends to B, B sends to C, C sends back to A) represent a high-priority fraud
indicator regardless of individual account risk scores. All participants in a detected ring
are automatically elevated to medium risk (minimum score 0.50) pending investigation.

4.3 Hub Accounts: Accounts receiving funds from two or more distinct sources and simultaneously
distributing to two or more distinct destinations are classified as hub accounts. Hub account
behavior combined with a customer risk score above 0.40 triggers a hub investigation flag.

4.4 Blast Radius Assessment: When a Critical Risk customer (score at or above 0.90) is
identified, the compliance team must map all accounts and customers reachable within two
transaction hops. This blast radius assessment determines the scope of potential contamination
and identifies which other customers may require immediate EDD review.
""",
        ),
        (
            "5. Risk Score Review and Appeals",
            """
5.1 Periodic Review: Risk scores are recalculated monthly for all customers. HRC customers
are reviewed weekly. Critical Risk customers are reviewed daily.

5.2 Advisor-Initiated Review: An advisor may request an expedited risk score review for any
client by submitting a Risk Review Request (RRR) form to the Risk Analytics team. Reviews
are completed within 10 business days.

5.3 Customer Appeals: Customers have the right to appeal their risk classification. Appeals
must be submitted in writing and are reviewed by the Customer Risk Appeals Committee within
30 days.

5.4 Regulatory Disclosure: Risk scores and HRC designations are not disclosed to customers
except in jurisdictions where local law requires disclosure.
""",
        ),
    ],
}

_ADVISOR_GUIDE = {
    "title": "Advisor Compliance Guidelines",
    "subtitle": "Advisor Relations and Compliance — Reference ARC-ACG-003 | Version 1.8",
    "sections": [
        (
            "1. Advisor Responsibilities Overview",
            """
Financial advisors are the first line of defense in detecting and preventing financial crime.
Advisors have direct relationships with clients and are uniquely positioned to identify unusual
behavior, inconsistencies in client profiles, and suspicious transaction patterns.

Core advisor compliance obligations:
- Know Your Customer (KYC): Maintain current and accurate knowledge of each client's financial
  situation, source of funds, and business activities.
- Transaction Awareness: Monitor your clients' account activity for patterns inconsistent with
  their known profile.
- Mandatory Reporting: Report any suspicious activity to the BSA/AML Compliance Officer
  immediately, regardless of transaction size.
- Confidentiality: Never disclose to a client that their account is under review or that a SAR
  has been filed.
""",
        ),
        (
            "2. High-Risk Client Management",
            """
Advisors managing High Risk Customers (HRC — risk score at or above 0.80) have heightened
obligations:

2.1 Onboarding Review: Before accepting a new HRC client, the advisor must obtain approval
from their branch compliance officer. A pre-approval Enhanced Due Diligence package must be
submitted.

2.2 Ongoing Monitoring: Advisors must review all transactions over $5,000 for HRC clients
within 48 hours. Unusual transactions must be flagged within the compliance workflow system.

2.3 Quarterly Reviews: Conduct a structured review meeting with each HRC client every quarter.
Document the meeting outcome, any changes to the client's financial profile, and any suspicious
indicators observed.

2.4 Network Awareness: Be aware that your HRC clients may have relationships with other
high-risk individuals. A cluster of HRC clients who transact with each other represents a
compounded risk that must be reported to compliance.

2.5 Critical Risk Clients (score at or above 0.90): These clients require immediate escalation
to the Chief Compliance Officer. The advisor must not take any action on behalf of a critical
risk client without prior written compliance approval.
""",
        ),
        (
            "3. Portfolio Concentration and Escalation",
            """
3.1 Portfolio Composition Monitoring: Advisors must be aware of the risk composition of their
managed portfolio. Quarterly portfolio risk reports are generated by the compliance system and
reviewed by the advisor's branch manager.

3.2 Concentration Thresholds:
- If more than 25% of an advisor's clients are HRC, the advisor must submit a portfolio
  concentration explanation to their compliance officer within 30 days.
- If more than 50% of an advisor's clients are HRC, an automatic escalation to the Head of
  Compliance is triggered and a portfolio restructuring plan must be submitted within 60 days.
- If 100% of an advisor's clients are HRC, this constitutes a critical compliance breach
  requiring immediate investigation. This scenario suggests either systematic misclassification
  or deliberate accumulation of high-risk clients, both of which require urgent review by the
  Chief Compliance Officer.

3.3 New Client Acceptance: Advisors must obtain pre-approval from compliance before onboarding
any client with an estimated risk score above 0.70. Advisors who already have more than 40%
HRC clients require Head of Compliance approval for any new HRC client.
""",
        ),
        (
            "4. Suspicious Activity Indicators",
            """
Advisors must be familiar with the following red flags and report them to compliance immediately:

4.1 Structuring Indicators:
- Client makes multiple deposits just below $10,000.
- Client requests transfers to multiple parties in amounts averaging below $5,000 following
  a large deposit above $10,000.
- Client shows knowledge of reporting thresholds or asks about cash reporting rules.
- Frequent round-number transactions designed to split a larger sum.

4.2 Circular Flow Indicators:
- Funds that appear to leave an account and return within a short period through intermediary
  accounts — for example Account A sends to B, B sends to C, C sends back to A.
- Multiple clients managed by the same advisor sending funds to each other in a loop.

4.3 Unexplained Wealth:
- Transaction amounts are inconsistent with the client's known income, occupation, or business
  activity.
- The client is unable or unwilling to explain the source of large deposits.
- A significant increase in account activity that is not explained by a known business event.

4.4 High-Risk Network Connections:
- A client is identified as a direct counterparty to a Critical Risk client (score at or above
  0.90).
- A low-risk client's account suddenly shows a high volume of transactions with accounts owned
  by HRC clients. This is the risk contagion pattern and requires immediate review.
""",
        ),
        (
            "5. Training and Certification Requirements",
            """
5.1 Mandatory Annual Training: All advisors must complete the Annual AML Compliance Training
course. Failure to complete training by the deadline results in temporary suspension of
client-facing duties.

5.2 Certification: Advisors must pass the AML Certification Exam (minimum score 80%) annually.
The exam covers: AML regulations, SAR filing procedures, EDD requirements, and suspicious
activity recognition.

5.3 Ongoing Education: Advisors managing HRC clients must additionally complete the High-Risk
Client Management module (4 hours) and the Network Risk Awareness workshop (2 hours) each year.

5.4 Breach Reporting: Advisors who become aware of compliance breaches — including those by
colleagues — must report them to the compliance hotline. Failure to report known breaches is
itself a compliance violation subject to disciplinary action.
""",
        ),
    ],
}


# ── Builder ───────────────────────────────────────────────────────────────────

def _build_pdf(path: Path, doc_data: dict) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.9 * inch,
        leftMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
    )
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.spaceAfter = 6
    body.leading = 14

    story = [
        Paragraph(doc_data["title"], styles["Title"]),
        Paragraph(doc_data["subtitle"], styles["Italic"]),
        HRFlowable(width="100%", thickness=1, color=colors.grey),
        Spacer(1, 0.2 * inch),
    ]
    for section_title, section_body in doc_data["sections"]:
        story.append(Paragraph(section_title, styles["Heading2"]))
        for para in section_body.strip().split("\n\n"):
            text = para.strip()
            if text:
                story.append(Paragraph(text, body))
        story.append(Spacer(1, 0.15 * inch))

    doc.build(story)


def generate_pdfs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("aml_policy.pdf", _AML_POLICY),
        ("customer_risk_framework.pdf", _RISK_FRAMEWORK),
        ("advisor_compliance_guide.pdf", _ADVISOR_GUIDE),
    ]
    for filename, data in specs:
        path = PDF_DIR / filename
        _build_pdf(path, data)
        print(f"Created {path}")
    print(f"\nGenerated {len(specs)} PDFs in {PDF_DIR}")


if __name__ == "__main__":
    generate_pdfs()
