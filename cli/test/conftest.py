import json
import os
from pathlib import Path

import boto3
import botocore.client
import pytest
from moto import mock_s3

from src.base import TextBlock


class S3Client:
    """Helper class to connect to S3 and perform actions on buckets and documents."""

    def __init__(self, region):
        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=botocore.client.Config(
                signature_version="s3v4",
                region_name=region,
                connect_timeout=10,
            ),
        )


@pytest.fixture
def s3_bucket_and_region() -> dict:
    return {
        "bucket": "test-bucket",
        "region": "eu-west-1",
    }


@pytest.fixture
def input_prefix() -> str:
    return "embeddings_input_test"


@pytest.fixture
def output_prefix() -> str:
    return "embeddings_output_test"


@pytest.fixture
def test_file_name() -> str:
    return "CCLWTEST.executive.1.1.json"


@pytest.fixture
def test_file_key(s3_bucket_and_region, input_prefix, test_file_name) -> str:
    return "embeddings_input_test/test.txt"


@pytest.fixture
def test_input_dir_s3(s3_bucket_and_region, input_prefix) -> str:
    return f"s3://{s3_bucket_and_region['bucket']}/{input_prefix}/"


@pytest.fixture
def test_output_dir_s3(s3_bucket_and_region, output_prefix) -> str:
    return f"s3://{s3_bucket_and_region['bucket']}/{output_prefix}/"


@pytest.fixture
def test_pdf_file_json() -> dict:
    return {
        "document_id": "test_pdf",
        "document_name": "test_pdf",
        "document_description": "test_pdf_description",
        "document_source_url": "https://cdn.climatepolicyradar.org/EUR/2013/EUR-2013-01-01-Overview+of+CAP+Reform"
        "+2014-2020_6237180d8c443d72c06c9167019ca177.pdf",
        "document_cdn_object": "EUR/2013/EUR-2013-01-01-Overview+of+CAP+Reform+2014"
        "-2020_6237180d8c443d72c06c9167019ca177.pdf",
        "document_md5_sum": "abcdefghijk",
        "languages": ["en"],
        "document_metadata": {
            "publication_ts": "2022-10-25 12:43:00.869045",
            "geography": "test_geo",
            "category": "test_category",
            "source": "test_source",
            "type": "test_type",
            "sectors": ["sector1", "sector2"],
        },
        "translated": False,
        "document_slug": "XYX",
        "document_content_type": "application/pdf",
        "html_data": None,
        "pdf_data": {
            "page_metadata": [
                {"page_number": 0, "dimensions": [596.0, 842.0]},
                {"page_number": 1, "dimensions": [596.0, 842.0]},
                {"page_number": 2, "dimensions": [596.0, 842.0]},
                {"page_number": 3, "dimensions": [596.0, 842.0]},
                {"page_number": 4, "dimensions": [596.0, 842.0]},
                {"page_number": 5, "dimensions": [596.0, 842.0]},
                {"page_number": 6, "dimensions": [596.0, 842.0]},
                {"page_number": 7, "dimensions": [596.0, 842.0]},
                {"page_number": 8, "dimensions": [596.0, 842.0]},
                {"page_number": 9, "dimensions": [596.0, 842.0]},
            ],
            "md5sum": "6237180d8c443d72c06c9167019ca177",
            "text_blocks": [
                {
                    "text": [
                        "Contact: DG Agriculture and\nRural Development, Unit for\nAgricultural Policy "
                        "Analysis\nand "
                        "Perspectives. "
                    ],
                    "text_block_id": "p_0_b_0",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.6339805126190186,
                    "coords": [
                        [10.998469352722168, 702.727294921875],
                        [134.93479919433594, 702.727294921875],
                        [134.93479919433594, 737.7978515625],
                        [10.998469352722168, 737.7978515625],
                    ],
                    "page_number": 0,
                },
                {
                    "text": [
                        "The new CAP maintains the two pillars, but increases the links\nbetween them, "
                        "thus offering "
                        "a more holistic and integrated approach\nto policy support. Specifically it introduces a "
                        "new architecture of\ndirect payments; better targeted, more equitable and greener, "
                        "an\nenhanced safety net and strengthened rural development. As a\nresult it is adapted to "
                        "meet the challenges ahead by being more\nefficient and contributing to a more competitive "
                        "and sustainable EU\nagriculture. This Brief gives an overview of the reform and "
                        'outlines\nthe "why and how" of the new CAP 2014-2020. '
                    ],
                    "text_block_id": "p_0_b_1",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9978094696998596,
                    "coords": [
                        [172.95388793945312, 442.6825256347656],
                        [567.635986328125, 442.6825256347656],
                        [567.635986328125, 560.8237915039062],
                        [172.95388793945312, 560.8237915039062],
                    ],
                    "page_number": 0,
                },
                {
                    "text": [
                        "The new agreement on CAP reform reached in 2013 is the fruit of\nthree years of reflection, "
                        "discussion and intensive negotiation. While\ncontinuing on the path of reform started in "
                        "the early '90's this deal is\nhistoric in many respects; for the first time the entire CAP "
                        "was\nreviewed all at once and the European Parliament acted as co-\nlegislator with the "
                        "Council. "
                    ],
                    "text_block_id": "p_0_b_2",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9603692293167114,
                    "coords": [
                        [179.08694458007812, 346.8304443359375],
                        [561.4661865234375, 346.8304443359375],
                        [561.4661865234375, 427.0174560546875],
                        [179.08694458007812, 427.0174560546875],
                    ],
                    "page_number": 0,
                },
                {
                    "text": [
                        "http://ec.europa.eu/agriculture/policy-perspectives/policy-briefs/index_en.htm\nThis Brief "
                        "does not necessarily represent the official views of the European "
                        "Commission.\nhttp://ec.europa.eu/agriculture/policy-perspectives/index_en.htm "
                    ],
                    "text_block_id": "p_0_b_3",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.6984515190124512,
                    "coords": [
                        [184.9188995361328, 755.02685546875],
                        [548.78369140625, 755.02685546875],
                        [548.78369140625, 782.5874633789062],
                        [184.9188995361328, 782.5874633789062],
                    ],
                    "page_number": 0,
                },
                {
                    "text": [
                        "The CAP reform started more than 3 years\nago in 2010 with a public debate, followed\nby "
                        "the publication of the Commission's\nCommunication on its vision of agriculture\nand the "
                        "challenges and priorities for the\nfuture CAP¹ and finally by legislative\nproposals for "
                        "the first ever overhaul of the\nentire policy. The decision-making process\ndiffered from "
                        "previous reforms, with the\nEuropean Parliament for the first time acting\nas co-legislator "
                        "with the Council. "
                    ],
                    "text_block_id": "p_1_b_0",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9975109100341797,
                    "coords": [
                        [25.807947158813477, 134.7040557861328],
                        [282.11480712890625, 134.7040557861328],
                        [282.11480712890625, 285.88555908203125],
                        [25.807947158813477, 285.88555908203125],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "It also took place in the framework of the\ndiscussions on the overall EU "
                        "budgetary\nframework for 2014-2020, the Multiannual\nFinancial Framework (MFF), "
                        "which provides\nfor the funds at the disposal of the EU\nincluding\nCAP. After "
                        "intensive\nnegotiations, in 2013 a deal was secured\nboth on the CAP and the MFF. The new "
                        "CAP\n2014-2020 agreed by the Council and the\nEuropean Parliament retains most of "
                        "the\nessential objectives\napproaches\nthe\nand\nproposed by the Commission, albeit with "
                        "a\nlower budget than proposed by the\nCommission.\n2.\nCHALLENGES & OBJECTIVES\nThe new CAP "
                        "builds on past reforms to\nmeet new challenges and objectives. "
                    ],
                    "text_block_id": "p_1_b_1",
                    "language": "en",
                    "type": "Ambiguous",
                    "type_confidence": 1.0,
                    "coords": [
                        [25.807947158813477, 285.88555908203125],
                        [282.11480712890625, 285.88555908203125],
                        [282.11480712890625, 581.1939697265625],
                        [25.807947158813477, 581.1939697265625],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "For more than twenty years, starting in\n1992, the CAP has been through successive\nreforms "
                        "which have increased market\norientation for agriculture while providing "
                    ],
                    "text_block_id": "p_1_b_2",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9925699830055237,
                    "coords": [
                        [27.3324031829834, 581.1939697265625],
                        [281.8106689453125, 581.1939697265625],
                        [281.8106689453125, 636.1080322265625],
                        [27.3324031829834, 636.1080322265625],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "It also took place in the framework of the\ndiscussions on the overall EU "
                        "budgetary\nframework for 2014-2020, the Multiannual\nFinancial Framework (MFF), "
                        "which provides\nfor the funds at the disposal of the EU\nincluding the CAP. After "
                        "intensive\nnegotiations, in 2013 a deal was secured\nboth on the CAP and the MFF. The new "
                        "CAP\n2014-2020 agreed by the Council and the\nEuropean Parliament retains most of "
                        "the\nessential objectives\napproaches\nand\nproposed by the Commission, albeit with "
                        "a\nlower budget than proposed by the\nCommission. "
                    ],
                    "text_block_id": "p_1_b_3",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9871504306793213,
                    "coords": [
                        [27.318716049194336, 293.22479248046875],
                        [280.6902160644531, 293.22479248046875],
                        [280.6902160644531, 482.4350280761719],
                        [27.318716049194336, 482.4350280761719],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "2. CHALLENGES & OBJECTIVES\nThe new CAP builds on past reforms to\nmeet new challenges and "
                        "objectives.\nFor more than twenty years, starting in\n1992, the CAP has been through "
                        "successive\nreforms which have increased market\norientation for agriculture while "
                        "providing\nCommission Commutention 00\nthe CAD towardo 2070 "
                    ],
                    "text_block_id": "p_1_b_4",
                    "language": "en",
                    "type": "Ambiguous",
                    "type_confidence": 1.0,
                    "coords": [
                        [27.318716049194336, 482.4350280761719],
                        [280.6902160644531, 482.4350280761719],
                        [280.6902160644531, 665.149658203125],
                        [27.318716049194336, 665.149658203125],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "* Commission Communication on the CAP towards 2020,\nCOM(2010) 672 final"
                    ],
                    "text_block_id": "p_1_b_5",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9585784077644348,
                    "coords": [
                        [28.513404846191406, 665.149658203125],
                        [279.2921142578125, 665.149658203125],
                        [279.2921142578125, 685.1968383789062],
                        [28.513404846191406, 685.1968383789062],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "² The Commission tabled four legislative proposals on Direct\nPayments, Rural Development, "
                        "the single Common Market\nOrganisation and horizontal aspects of the CAP, "
                        "based on an\nImpact Assessment and extensive consultation with citizens\nand stakeholders. "
                    ],
                    "text_block_id": "p_1_b_6",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9977612495422363,
                    "coords": [
                        [28.13901710510254, 697.1915893554688],
                        [278.33380126953125, 697.1915893554688],
                        [278.33380126953125, 746.2515258789062],
                        [28.13901710510254, 746.2515258789062],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "The new CAP builds on past reforms to\nmeet new challenges and objectives."
                    ],
                    "text_block_id": "p_1_b_7",
                    "language": "en",
                    "type": "Title",
                    "type_confidence": 0.7103927731513977,
                    "coords": [
                        [34.05950164794922, 542.3276977539062],
                        [275.6168212890625, 542.3276977539062],
                        [275.6168212890625, 569.9240112304688],
                        [34.05950164794922, 569.9240112304688],
                    ],
                    "page_number": 1,
                },
                {
                    "text": ["CHALLENGES & OBJECTIVES"],
                    "text_block_id": "p_1_b_8",
                    "language": "en",
                    "type": "Title",
                    "type_confidence": 0.5244852900505066,
                    "coords": [
                        [55.747737884521484, 516.7940673828125],
                        [232.61456298828125, 516.7940673828125],
                        [232.61456298828125, 531.4217529296875],
                        [55.747737884521484, 531.4217529296875],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "income support and safety net mechanisms\nfor producers, improved the integration "
                        "of\nenvironmental requirements and reinforced\nsupport for rural development across the EU. "
                    ],
                    "text_block_id": "p_1_b_9",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9015650749206543,
                    "coords": [
                        [312.5420837402344, 109.67631530761719],
                        [569.4307861328125, 109.67631530761719],
                        [569.4307861328125, 162.09609985351562],
                        [312.5420837402344, 162.09609985351562],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "The new policy continues along this reform\npath, moving from product to producer\nsupport "
                        "and now to a more land-based\napproach. This is in\nin response to the\nchallenges facing "
                        "the sector, many of which\nare driven by factors that are external to\nagriculture. "
                    ],
                    "text_block_id": "p_1_b_10",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9820030927658081,
                    "coords": [
                        [311.6417541503906, 176.51788330078125],
                        [567.279052734375, 176.51788330078125],
                        [567.279052734375, 268.0145568847656],
                        [311.6417541503906, 268.0145568847656],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "These have been identified as economic\n(including food security and globalisation, "
                        "a\ndeclining rate of productivity growth, price\nvolatility, pressures on production costs "
                        "due\nto high input prices and the deteriorating\nposition of farmers in the food supply "
                        "chain),\nenvironmental (relating to resource\nefficiency, soil and water quality and "
                        "threats\nto habitats and biodiversity) and territorial\n(where\nare faced "
                        "with\ndemographic,\nand social\nrural\nareas\neconomic\ndevelopments including depopulation "
                        "and\nrelocation of businesses). "
                    ],
                    "text_block_id": "p_1_b_11",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9854642748832703,
                    "coords": [
                        [313.46038818359375, 282.8002014160156],
                        [568.2947998046875, 282.8002014160156],
                        [568.2947998046875, 456.5018615722656],
                        [313.46038818359375, 456.5018615722656],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "Since the role of the CAP is to provide a\npolicy framework that supports and\nencourages "
                        "producers to address these\nchallenges while remaining coherent with\nother EU policies, "
                        "this translates into three\nlong-term CAP objectives: viable food\nproduction, sustainable "
                        "management of\nnatural resources and climate action and\nbalanced territorial development. "
                    ],
                    "text_block_id": "p_1_b_12",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9497474431991577,
                    "coords": [
                        [314.9910888671875, 466.8597412109375],
                        [570.7342529296875, 466.8597412109375],
                        [570.7342529296875, 584.8880615234375],
                        [314.9910888671875, 584.8880615234375],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "To achieve these long-term goals, the\nhad to be adapted.\ntherefore focused on "
                        "the\nexisting CAP instruments\nThe reform\noperational objectives of delivering "
                        "more\neffective policy instruments, designed to\nimprove the competitiveness of "
                        "the\nagricultural sector and its sustainability over\nthe long term (Chart 1). "
                    ],
                    "text_block_id": "p_1_b_13",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9796818494796753,
                    "coords": [
                        [313.8314514160156, 597.879150390625],
                        [571.148193359375, 597.879150390625],
                        [571.148193359375, 703.856201171875],
                        [313.8314514160156, 703.856201171875],
                    ],
                    "page_number": 1,
                },
                {
                    "text": [
                        "In short, EU agriculture needs to attain\nhigher levels of production of safe and\nquality "
                        "food, while preserving the natural\nresources that agricultural productivity\ndepends upon. "
                    ],
                    "text_block_id": "p_2_b_0",
                    "language": "en",
                    "type": "Title",
                    "type_confidence": 0.577865481376648,
                    "coords": [
                        [26.7734375, 313.9053039550781],
                        [281.1876525878906, 313.9053039550781],
                        [281.1876525878906, 380.9349365234375],
                        [26.7734375, 380.9349365234375],
                    ],
                    "page_number": 2,
                },
                {
                    "text": [
                        "This can only be achieved by a competitive\nand viable agricultural sector "
                        "operating\nwithin a properly functioning supply chain\nand which contributes to the "
                        "maintenance of\na thriving rural economy. In addition, to\nachieve these long-term goals, "
                        "better\ntargeting of the available CAP budget will be\nneeded. "
                    ],
                    "text_block_id": "p_2_b_1",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.985622227191925,
                    "coords": [
                        [27.018037796020508, 394.8108825683594],
                        [283.1033630371094, 394.8108825683594],
                        [283.1033630371094, 503.0022277832031],
                        [27.018037796020508, 503.0022277832031],
                    ],
                    "page_number": 2,
                },
                {
                    "text": ["Source: DG Agriculture and Rural Development."],
                    "text_block_id": "p_2_b_2",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9863702058792114,
                    "coords": [
                        [27.249122619628906, 281.004150390625],
                        [222.2008514404297, 281.004150390625],
                        [222.2008514404297, 291.407470703125],
                        [27.249122619628906, 291.407470703125],
                    ],
                    "page_number": 2,
                },
                {
                    "text": ["What amounts will be available for the\nnew CAP?"],
                    "text_block_id": "p_2_b_3",
                    "language": "en",
                    "type": "Title",
                    "type_confidence": 0.7921167016029358,
                    "coords": [
                        [315.7379455566406, 133.61001586914062],
                        [564.5430297851562, 133.61001586914062],
                        [564.5430297851562, 161.7198944091797],
                        [315.7379455566406, 161.7198944091797],
                    ],
                    "page_number": 2,
                },
                {
                    "text": [
                        "The amounts for the CAP agreed under the\nnew EU multiannual financial framework "
                        "for\n2014-2020 are outlined in the table\nbelow. The Commission had proposed that,"
                        "\nin nominal terms, the amounts for both\npillars of the CAP for 2014-2020 would be\nfrozen "
                        "at the level of 2013. In real terms\nCAP funding will decrease compared to the\ncurrent "
                        "period. Compared to the\nCommission proposal, the amount for pillar 1\nwas cut by 1.8% and "
                        "for pillar 2 by 7.6% (in\n2011 prices). "
                    ],
                    "text_block_id": "p_2_b_4",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9895569682121277,
                    "coords": [
                        [314.5663757324219, 173.60450744628906],
                        [569.4752197265625, 173.60450744628906],
                        [569.4752197265625, 336.28778076171875],
                        [314.5663757324219, 336.28778076171875],
                    ],
                    "page_number": 2,
                },
                {
                    "text": [
                        "This means a total amount of EUR 362.787\nbillion for 2014-2020, of which EUR "
                        "277.851\nbillion is foreseen for Direct Payments and\nmarket-related expenditure (Pillar 1) "
                        "and\nEUR 84.936 billion for Rural Development\n(Pillar 2) in 2011 prices. Yet, "
                        "within the\ncurrent economic and financial climate,\nthese amounts within the MFF "
                        "show\ncontinued strong support for an ambitious\nagricultural policy which represents 37.8% "
                        "of\nthe entire ceiling for the period 2014-2020. "
                    ],
                    "text_block_id": "p_2_b_5",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9928650856018066,
                    "coords": [
                        [312.7943420410156, 346.2590637207031],
                        [569.8253173828125, 346.2590637207031],
                        [569.8253173828125, 490.6370849609375],
                        [312.7943420410156, 490.6370849609375],
                    ],
                    "page_number": 2,
                },
                {
                    "text": ["Source: DG Agriculture and Rural Development"],
                    "text_block_id": "p_2_b_6",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9880040884017944,
                    "coords": [
                        [314.1606140136719, 654.7015991210938],
                        [507.0452880859375, 654.7015991210938],
                        [507.0452880859375, 664.351806640625],
                        [314.1606140136719, 664.351806640625],
                    ],
                    "page_number": 2,
                },
                {
                    "text": [
                        "The radical change in the orientation of the\nCAP is demonstrated by the evolution "
                        "of\nexpenditure, echoing the policy shift since\n1992³, away from product based "
                        "support\ntowards producer support and considerations\nfor the environment (chart 2). "
                    ],
                    "text_block_id": "p_3_b_0",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9974133372306824,
                    "coords": [
                        [25.84006118774414, 151.5837860107422],
                        [280.8387756347656, 151.5837860107422],
                        [280.8387756347656, 232.2913818359375],
                        [25.84006118774414, 232.2913818359375],
                    ],
                    "page_number": 3,
                },
                {
                    "text": [
                        "In 1992 market management represented\nover 90% of total CAP expenditure, driven\nby export "
                        "refunds and intervention\npurchases. By the end of 2013 it dropped to\njust 5% as market "
                        "intervention has become\na safety net tool for times of crisis and direct\npayments are the "
                        "major source of support;\n94% of which are decoupled from\nproduction. "
                    ],
                    "text_block_id": "p_3_b_1",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9144823551177979,
                    "coords": [
                        [24.573030471801758, 247.81967163085938],
                        [280.61480712890625, 247.81967163085938],
                        [280.61480712890625, 356.9373779296875],
                        [24.573030471801758, 356.9373779296875],
                    ],
                    "page_number": 3,
                },
                {
                    "text": [
                        "production.\nin billion EUR current "
                        "prices\n70\n60\n50\n40\n30\n20\n10\nEU-12\nCAD\n1996\n4661\nExport refunds\nCoupled direct "
                        "payments\nPossible flexibility for RD\nSource: DG Agriculture and Rural "
                        "Development\n8661\nEU-15\n1000\nOther ma\nDecouple\nPossible\n7017 "
                    ],
                    "text_block_id": "p_3_b_2",
                    "language": "en",
                    "type": "Ambiguous",
                    "type_confidence": 1.0,
                    "coords": [
                        [24.573030471801758, 356.9373779296875],
                        [284.76953125, 356.9373779296875],
                        [284.76953125, 739.556640625],
                        [24.573030471801758, 739.556640625],
                    ],
                    "page_number": 3,
                },
                {
                    "text": [
                        "³ Chart 2 shows CAP actual payments from 1990- 2012,\ncommitments for 2013 and the new MFF "
                        "ceiling from 2014-\n2020. "
                    ],
                    "text_block_id": "p_3_b_3",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9739090800285339,
                    "coords": [
                        [26.489742279052734, 739.556640625],
                        [284.76953125, 739.556640625],
                        [284.76953125, 769.3900146484375],
                        [26.489742279052734, 769.3900146484375],
                    ],
                    "page_number": 3,
                },
                {
                    "text": ["Source: DG Agriculture and Rural Development"],
                    "text_block_id": "p_3_b_4",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.9940191507339478,
                    "coords": [
                        [26.59664535522461, 705.384765625],
                        [220.10818481445312, 705.384765625],
                        [220.10818481445312, 715.4427490234375],
                        [26.59664535522461, 715.4427490234375],
                    ],
                    "page_number": 3,
                },
                {
                    "text": [
                        "Other market support\nMarket support\nNew direct payments\nDecoupled direct "
                        "payments\nPossible flexibility for direct payments Rural development (RD) "
                    ],
                    "text_block_id": "p_3_b_5",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 0.8384557366371155,
                    "coords": [
                        [249.6709442138672, 679.8251953125],
                        [501.9338073730469, 679.8251953125],
                        [501.9338073730469, 700.936767578125],
                        [249.6709442138672, 700.936767578125],
                    ],
                    "page_number": 3,
                },
                {
                    "text": ["AND"],
                    "text_block_id": "p_3_b_6",
                    "language": None,
                    "type": "Title",
                    "type_confidence": 0.9026237726211548,
                    "coords": [
                        [252.4384307861328, 110.40703582763672],
                        [279.9317321777344, 110.40703582763672],
                        [279.9317321777344, 122.95552062988281],
                        [252.4384307861328, 122.95552062988281],
                    ],
                    "page_number": 3,
                },
            ],
        },
    }


@pytest.fixture
def test_no_content_type_file_json() -> dict:
    return {
        "document_id": "test_no_content_type",
        "document_name": "test_no_content_type",
        "document_description": "test_description",
        "document_source_url": None,
        "document_cdn_object": None,
        "document_md5_sum": None,
        "document_metadata": {
            "publication_ts": "2022-10-25 12:45:00.869045",
            "geography": "test_geo",
            "category": "test_category",
            "source": "test_source",
            "type": "test_type",
            "sectors": ["sector1", "sector2"],
        },
        "languages": None,
        "translated": False,
        "document_slug": "YYY",
        "document_content_type": None,
        "html_data": None,
        "pdf_data": None,
    }


@pytest.fixture
def test_html_file_json() -> dict:
    return {
        "document_id": "test_html",
        "document_name": "test_html",
        "document_description": "test_html_description",
        "document_source_url": "https://www.industry.gov.au/funding-and-incentives/emissions-reduction-fund",
        "document_cdn_object": None,
        "document_md5_sum": None,
        "languages": ["en"],
        "document_metadata": {
            "publication_ts": "2022-10-25 12:43:00.869045",
            "geography": "test_geo",
            "category": "test_category",
            "source": "test_source",
            "type": "test_type",
            "sectors": ["sector1", "sector2"],
        },
        "translated": False,
        "document_slug": "YYY",
        "document_content_type": "text/html",
        "html_data": {
            "detected_title": "Machinery of Government (MoG) changes to our department from 1 July 2022",
            "detected_date": "2020-10-22",
            "has_valid_text": False,
            "text_blocks": [
                {
                    "text": [
                        "From 1 July 2022, the Department of Industry, Science, Energy and Resources (DISER) "
                        "becomes the Department of Industry, Science and Resources (DISR). This follows "
                        "Administrative Arrangements Orders issued on 1 June 2022. "
                    ],
                    "text_block_id": "b0",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Department functions"],
                    "text_block_id": "b1",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "The climate change and energy functions that previously sat with our department have "
                        "transferred to the new Department of Climate Change, Energy, the Environment and Water "
                        "(DCCEW). "
                    ],
                    "text_block_id": "b2",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "The Department of Industry, Science and Resources retains the industry, science and "
                        "resources functions. It also takes on several functions that previously sat with the "
                        "department of Prime Minister and Cabinet. "
                    ],
                    "text_block_id": "b3",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "Our department’s organisational chart outlines the new structure and key staff changes."
                    ],
                    "text_block_id": "b4",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "Visit the new Department of Climate Change, Energy, the Environment and Water "
                        "dcceew.gov.au website to learn more about its remit and structure. "
                    ],
                    "text_block_id": "b5",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Ministers"],
                    "text_block_id": "b6",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "The Minister for Climate Change and Energy and Assistant Minister for Climate Change "
                        "and Energy will be responsible for these portfolios in the Department of Climate "
                        "Change, Energy, the Environment and Water. "
                    ],
                    "text_block_id": "b7",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "Visit the new minister.dcceew.gov.au website to find their media releases, speeches and "
                        "transcripts. "
                    ],
                    "text_block_id": "b8",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "The Minister for Industry and Science, Minister for Resources, Assistant Minister for "
                        "Manufacturing and Assistant Minister for Trade will stay with our department. "
                    ],
                    "text_block_id": "b9",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "Visit the current minister.industry.gov.au website to find their media releases, "
                        "speeches and transcripts. "
                    ],
                    "text_block_id": "b10",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Portfolio bodies and offices"],
                    "text_block_id": "b11",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "A number of portfolio bodies and offices also transfer as a result of these MoG changes."
                    ],
                    "text_block_id": "b12",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "Agencies or entities moving from the Department of Industry, Science, Energy and "
                        "Resources to the Department of Climate Change, Energy, the Environment and Water are: "
                    ],
                    "text_block_id": "b13",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "Offices or taskforces moving from the Department of the Prime Minister and Cabinet to "
                        "the Department of Industry, Science and Resources are: "
                    ],
                    "text_block_id": "b14",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Accessing website content"],
                    "text_block_id": "b15",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "Emissions reduction and energy content previously on industry.gov.au has transferred to "
                        "dcceew.gov.au. Consultations currently on consult.industry.gov.au will move later. "
                    ],
                    "text_block_id": "b16",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["You can find all related content via the links below."],
                    "text_block_id": "b17",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Climate change content"],
                    "text_block_id": "b18",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Policies and initiatives:"],
                    "text_block_id": "b19",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Regulations and standards:"],
                    "text_block_id": "b20",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Offices and committees:"],
                    "text_block_id": "b21",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Publications:"],
                    "text_block_id": "b22",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["News:"],
                    "text_block_id": "b23",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Consultations:"],
                    "text_block_id": "b24",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Energy content"],
                    "text_block_id": "b25",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Policies and initiatives:"],
                    "text_block_id": "b26",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Regulations and standards:"],
                    "text_block_id": "b27",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Committees"],
                    "text_block_id": "b28",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Publications:"],
                    "text_block_id": "b29",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["News:"],
                    "text_block_id": "b30",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Consultations:"],
                    "text_block_id": "b31",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Finding social media channels"],
                    "text_block_id": "b32",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "These social media channels are now part of the Department of Climate Change, Energy, "
                        "the Environment and Water: "
                    ],
                    "text_block_id": "b33",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "During the transition, you can still find climate change and energy videos on our "
                        "department’s YouTube channel. "
                    ],
                    "text_block_id": "b34",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": ["Signing up to newsletters"],
                    "text_block_id": "b35",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "These newsletters are now part of the Department of Climate Change, Energy, "
                        "the Environment and Water: "
                    ],
                    "text_block_id": "b36",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
                {
                    "text": [
                        "Over time we will transfer any newsletters, signup forms and subscriber lists we "
                        "currently host to the new department. "
                    ],
                    "text_block_id": "b37",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0,
                },
            ],
        },
        "pdf_data": None,
    }


@pytest.fixture
def pipeline_s3_objects_main(
    test_file_key,
    test_html_file_json,
):
    """
    Return a dict of s3 objects to be used in the pipeline s3 client fixture.

    This sets up a s3 bucket with the following objects:
    - Test parser output document

    Thus, we have a document that embeddings can be generated from in s3.
    """
    return {
        test_file_key: bytes(json.dumps(test_html_file_json).encode("UTF-8")),
    }


@pytest.fixture
def pipeline_s3_client_main(s3_bucket_and_region, pipeline_s3_objects_main):
    with mock_s3():
        s3_client = S3Client(s3_bucket_and_region["region"])

        s3_client.client.create_bucket(
            Bucket=s3_bucket_and_region["bucket"],
            CreateBucketConfiguration={
                "LocationConstraint": s3_bucket_and_region["region"]
            },
        )

        for key in pipeline_s3_objects_main:
            s3_client.client.put_object(
                Bucket=s3_bucket_and_region["bucket"],
                Key=key,
                Body=pipeline_s3_objects_main[key],
            )

        yield s3_client


def get_text_block(text_block_type: str) -> TextBlock:
    """Returns a TextBlock object with the given type."""
    return TextBlock(
        text=["test_text"],
        text_block_id="test_text_block_id",
        language="test_language",
        type=text_block_type,
        type_confidence=1.0,
        coords=[(0, 0), (0, 0), (0, 0), (0, 0)],
        page_number=0,
    )
