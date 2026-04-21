"""
Add Constitutional Articles to statutes.jsonl
Covers the ~30 most frequently cited articles in Indian court judgments.
"""

import json
import os

ARTICLES = [
    {
        "law": "CONSTITUTION",
        "section": "Article_14",
        "title": "Right to Equality",
        "text": "Article 14 — Equality before law: The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India. Prohibition of discrimination on grounds of religion, race, caste, sex or place of birth.\n\nArticle 14 in Simple Words\nEveryone is equal before the law. The government cannot treat people differently based on irrelevant factors. This is a fundamental right that guarantees equal treatment and prohibits arbitrary discrimination."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_15",
        "title": "Prohibition of Discrimination",
        "text": "Article 15 — Prohibition of discrimination on grounds of religion, race, caste, sex or place of birth: (1) The State shall not discriminate against any citizen on grounds only of religion, race, caste, sex, place of birth or any of them. (2) No citizen shall, on grounds only of religion, race, caste, sex, place of birth or any of them, be subject to any disability, liability, restriction or condition with regard to access to shops, public restaurants, hotels and places of public entertainment, or the use of wells, tanks, bathing ghats, roads and places of public resort maintained wholly or partly out of State funds or dedicated to the use of the general public.\n\nArticle 15 in Simple Words\nThe government cannot discriminate against citizens based on religion, race, caste, sex, or place of birth. All citizens have equal access to public places and facilities."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_19",
        "title": "Protection of Certain Rights Regarding Freedom of Speech",
        "text": "Article 19 — Protection of certain rights regarding freedom of speech, etc.: (1) All citizens shall have the right— (a) to freedom of speech and expression; (b) to assemble peaceably and without arms; (c) to form associations or unions; (d) to move freely throughout the territory of India; (e) to reside and settle in any part of the territory of India; (g) to practise any profession, or to carry on any occupation, trade or business. (2) Nothing in sub-clause (a) shall affect the operation of any existing law or prevent the State from making any law, in so far as such law imposes reasonable restrictions on the exercise of the right conferred by the said sub-clause in the interests of the sovereignty and integrity of India, the security of the State, friendly relations with foreign States, public order, decency or morality or in relation to contempt of court, defamation or incitement to an offence.\n\nArticle 19 in Simple Words\nCitizens have fundamental freedoms: speech, assembly, association, movement, residence, and profession. However, the State can impose reasonable restrictions in the interest of sovereignty, security, public order, decency, morality, contempt of court, or defamation."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_20",
        "title": "Protection in Respect of Conviction for Offences",
        "text": "Article 20 — Protection in respect of conviction for offences: (1) No person shall be convicted of any offence except for violation of the law in force at the time of the commission of the act charged as an offence, nor be subjected to a penalty greater than that which might have been inflicted under the law in force at the time of the commission of the offence. (2) No person shall be prosecuted and punished for the same offence more than once (Double Jeopardy). (3) No person accused of any offence shall be compelled to be a witness against himself (Right against Self-Incrimination).\n\nArticle 20 in Simple Words\nNo retroactive criminal laws, no double jeopardy (being punished twice for the same crime), and no one can be forced to testify against themselves."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_21",
        "title": "Protection of Life and Personal Liberty",
        "text": "Article 21 — Protection of life and personal liberty: No person shall be deprived of his life or personal liberty except according to procedure established by law. This article has been expansively interpreted by the Supreme Court to include right to live with dignity, right to livelihood, right to health, right to pollution-free environment, right to education, right to privacy, right to shelter, right to speedy trial, right to free legal aid, and right against solitary confinement.\n\nArticle 21 in Simple Words\nNo person can be deprived of their life or personal freedom except through a fair legal procedure. The Supreme Court has expanded this to cover dignity, livelihood, health, privacy, education, shelter, and many more rights essential for a dignified life."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_21A",
        "title": "Right to Education",
        "text": "Article 21A — Right to education: The State shall provide free and compulsory education to all children of the age of six to fourteen years in such manner as the State may, by law, determine. (Inserted by the Constitution (Eighty-sixth Amendment) Act, 2002).\n\nArticle 21A in Simple Words\nEvery child between ages 6 and 14 has a fundamental right to free and compulsory education provided by the State."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_22",
        "title": "Protection Against Arrest and Detention",
        "text": "Article 22 — Protection against arrest and detention in certain cases: (1) No person who is arrested shall be detained in custody without being informed, as soon as may be, of the grounds for such arrest nor shall he be denied the right to consult, and to be defended by, a legal practitioner of his choice. (2) Every person who is arrested and detained in custody shall be produced before the nearest magistrate within a period of twenty-four hours of such arrest excluding the time necessary for the journey from the place of arrest to the court of the magistrate and no such person shall be detained in custody beyond the said period without the authority of a magistrate.\n\nArticle 22 in Simple Words\nAn arrested person must be told why they were arrested, allowed to consult a lawyer, and must be produced before a magistrate within 24 hours. No detention beyond 24 hours without a magistrate's order."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_25",
        "title": "Freedom of Conscience and Free Profession, Practice and Propagation of Religion",
        "text": "Article 25 — Freedom of conscience and free profession, practice and propagation of religion: (1) Subject to public order, morality and health and to the other provisions of this Part, all persons are equally entitled to freedom of conscience and the right freely to profess, practise and propagate religion. (2) Nothing in this article shall affect the operation of any existing law or prevent the State from making any law regulating or restricting any economic, financial, political or other secular activity which may be associated with religious practice.\n\nArticle 25 in Simple Words\nEveryone has the right to freely practice, profess and propagate their religion, subject to public order, morality and health. The State can regulate secular activities associated with religious practice."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_32",
        "title": "Remedies for Enforcement of Fundamental Rights",
        "text": "Article 32 — Remedies for enforcement of rights conferred by this Part: (1) The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed. (2) The Supreme Court shall have power to issue directions or orders or writs, including writs in the nature of habeas corpus, mandamus, prohibition, quo warranto and certiorari, whichever may be appropriate, for the enforcement of any of the rights conferred by this Part. Dr. B.R. Ambedkar called Article 32 the 'heart and soul' of the Constitution.\n\nArticle 32 in Simple Words\nCitizens can directly approach the Supreme Court to enforce their fundamental rights. The Court can issue writs like habeas corpus, mandamus, prohibition, quo warranto, and certiorari. This is considered the most important article of the Constitution."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_44",
        "title": "Uniform Civil Code",
        "text": "Article 44 — Uniform civil code for the citizens: The State shall endeavour to secure for the citizens a uniform civil code throughout the territory of India. This is a Directive Principle of State Policy.\n\nArticle 44 in Simple Words\nThe State should work towards creating one set of civil laws (marriage, divorce, inheritance, adoption) applicable to all citizens regardless of religion. This is a directive principle, not directly enforceable."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_51A",
        "title": "Fundamental Duties",
        "text": "Article 51A — Fundamental duties: It shall be the duty of every citizen of India— (a) to abide by the Constitution and respect its ideals and institutions, the National Flag and the National Anthem; (b) to cherish and follow the noble ideals which inspired our national struggle for freedom; (c) to uphold and protect the sovereignty, unity and integrity of India; (d) to defend the country and render national service when called upon to do so; (e) to promote harmony and the spirit of common brotherhood amongst all the people of India transcending religious, linguistic and regional or sectional diversities; to renounce practices derogatory to the dignity of women; (f) to value and preserve the rich heritage of our composite culture; (g) to protect and improve the natural environment including forests, lakes, rivers and wild life, and to have compassion for living creatures; (h) to develop the scientific temper, humanism and the spirit of inquiry and reform; (i) to safeguard public property and to abjure violence; (j) to strive towards excellence in all spheres of individual and collective activity so that the nation constantly rises to higher levels of endeavour and achievement; (k) who is a parent or guardian to provide opportunities for education to his child between the age of six and fourteen years.\n\nArticle 51A in Simple Words\nEvery citizen has fundamental duties including respecting the Constitution, promoting harmony, protecting the environment, developing scientific temper, safeguarding public property, and providing education to children."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_72",
        "title": "Power of President to Grant Pardons",
        "text": "Article 72 — Power of President to grant pardons, etc., and to suspend, remit or commute sentences in certain cases: (1) The President shall have the power to grant pardons, reprieves, respites or remissions of punishment or to suspend, remit or commute the sentence of any person convicted of any offence— (a) in all cases where the punishment or sentence is by a Court Martial; (b) in all cases where the punishment or sentence is for an offence against any law relating to a matter to which the executive power of the Union extends; (c) in all cases where the sentence is a sentence of death.\n\nArticle 72 in Simple Words\nThe President of India can grant pardons, reduce or commute sentences of convicted persons, especially in court martial cases, Union law offences, and death sentences."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_136",
        "title": "Special Leave to Appeal by the Supreme Court",
        "text": "Article 136 — Special leave to appeal by the Supreme Court: (1) Notwithstanding anything in this Chapter, the Supreme Court may, in its discretion, grant special leave to appeal from any judgment, decree, determination, sentence or order in any cause or matter passed or made by any court or tribunal in the territory of India. (2) Nothing in clause (1) shall apply to any judgment, determination, sentence or order passed or made by any court or tribunal constituted by or under any law relating to the Armed Forces.\n\nArticle 136 in Simple Words\nThe Supreme Court has discretionary power to grant special leave to appeal against any judgment from any court or tribunal in India, except military tribunals. This is one of the most powerful provisions giving the Supreme Court wide appellate jurisdiction."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_141",
        "title": "Law Declared by Supreme Court to be Binding",
        "text": "Article 141 — Law declared by Supreme Court to be binding on all courts: The law declared by the Supreme Court shall be binding on all courts within the territory of India.\n\nArticle 141 in Simple Words\nAny law declared or interpreted by the Supreme Court is binding on all courts in India. This establishes the doctrine of precedent in Indian jurisprudence."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_142",
        "title": "Enforcement of Decrees and Orders of Supreme Court",
        "text": "Article 142 — Enforcement of decrees and orders of Supreme Court and orders as to discovery, etc.: (1) The Supreme Court in the exercise of its jurisdiction may pass such decree or make such order as is necessary for doing complete justice in any cause or matter pending before it, and any decree so passed or order so made shall be enforceable throughout the territory of India in such manner as may be prescribed by or under any law made by Parliament.\n\nArticle 142 in Simple Words\nThe Supreme Court has extraordinary power to pass any order necessary for 'complete justice' in any case before it, and such orders are enforceable across India. This is an extraordinary power used in landmark cases."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_226",
        "title": "Power of High Courts to Issue Certain Writs",
        "text": "Article 226 — Power of High Courts to issue certain writs: (1) Notwithstanding anything in Article 32, every High Court shall have powers, throughout the territories in relation to which it exercises jurisdiction, to issue to any person or authority, including in appropriate cases, any Government, within those territories directions, orders or writs, including writs in the nature of habeas corpus, mandamus, prohibition, quo warranto and certiorari, or any of them, for the enforcement of any of the rights conferred by Part III and for any other purpose.\n\nArticle 226 in Simple Words\nHigh Courts can issue writs (habeas corpus, mandamus, prohibition, quo warranto, certiorari) to enforce fundamental rights AND for any other purpose. This is broader than Article 32, which is limited to fundamental rights only."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_227",
        "title": "Power of Superintendence over All Courts by High Court",
        "text": "Article 227 — Power of superintendence over all courts by the High Court: (1) Every High Court shall have superintendence over all courts and tribunals throughout the territories in relation to which it exercises jurisdiction. (2) Without prejudice to the generality of the foregoing provision, the High Court may call for returns from such courts, make and issue general rules and prescribe forms for regulating the practice and proceedings of such courts.\n\nArticle 227 in Simple Words\nEvery High Court has supervisory power over all courts and tribunals within its jurisdiction. It can call for information, issue rules, and regulate procedures of lower courts."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_300A",
        "title": "Right to Property",
        "text": "Article 300A — Persons not to be deprived of property save by authority of law: No person shall be deprived of his property save by authority of law. (Inserted by the Constitution (Forty-fourth Amendment) Act, 1978, which removed right to property from the list of fundamental rights and made it a constitutional right.)\n\nArticle 300A in Simple Words\nNo person can be deprived of their property except by following proper legal procedures. While no longer a fundamental right, it remains a constitutional right that cannot be violated without legal authority."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_311",
        "title": "Dismissal, Removal or Reduction in Rank of Civil Servants",
        "text": "Article 311 — Dismissal, removal or reduction in rank of persons employed in civil capacities under the Union or a State: (1) No person who is a member of a civil service of the Union or an all-India service or a civil service of a State or holds a civil post under the Union or a State shall be dismissed or removed by an authority subordinate to that by which he was appointed. (2) No such person shall be dismissed or removed or reduced in rank except after an inquiry in which he has been informed of the charges against him and given a reasonable opportunity of being heard in respect of those charges.\n\nArticle 311 in Simple Words\nGovernment employees cannot be dismissed by an authority lower than the one that appointed them. Before dismissal, they must be given a proper inquiry with notice of charges and opportunity to be heard."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_352",
        "title": "Proclamation of Emergency",
        "text": "Article 352 — Proclamation of Emergency: (1) If the President is satisfied that a grave emergency exists whereby the security of India or of any part of the territory thereof is threatened, whether by war or external aggression or armed rebellion, he may, by Proclamation, make a declaration to that effect in respect of the whole of India or of such part of the territory thereof as may be specified in the Proclamation.\n\nArticle 352 in Simple Words\nThe President can declare a national emergency if India's security is threatened by war, external aggression, or armed rebellion. This emergency affects the entire country or specific parts."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_356",
        "title": "President's Rule — Failure of Constitutional Machinery in States",
        "text": "Article 356 — Provisions in case of failure of constitutional machinery in States: (1) If the President, on receipt of a report from the Governor of the State or otherwise, is satisfied that a situation has arisen in which the government of the State cannot be carried on in accordance with the provisions of this Constitution, the President may by Proclamation assume to himself all or any of the functions of the Government of the State.\n\nArticle 356 in Simple Words\nIf a state government cannot function according to the Constitution, the President can impose President's Rule — taking over the state's governance. This is commonly used when there is political instability in a state."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_368",
        "title": "Power of Parliament to Amend the Constitution",
        "text": "Article 368 — Power of Parliament to amend the Constitution and procedure therefor: (1) Notwithstanding anything in this Constitution, Parliament may in exercise of its constituent power amend by way of addition, variation or repeal any provision of this Constitution in accordance with the procedure laid down in this article. (2) An amendment of this Constitution may be initiated only by the introduction of a Bill for the purpose in either House of Parliament, and when the Bill is passed in each House by a majority of the total membership of that House and by a majority of not less than two-thirds of the members of that House present and voting, it shall be presented to the President who shall give his assent to the Bill and thereupon the Constitution shall stand amended.\n\nArticle 368 in Simple Words\nParliament can amend the Constitution through a special majority (2/3rds of members present and voting + majority of total membership). Some amendments also need ratification by half the state legislatures. The basic structure of the Constitution cannot be amended."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_12",
        "title": "Definition of State",
        "text": "Article 12 — Definition: In this Part, unless the context otherwise requires, 'the State' includes the Government and Parliament of India and the Government and the Legislature of each of the States and all local or other authorities within the territory of India or under the control of the Government of India.\n\nArticle 12 in Simple Words\nThe term 'State' in the context of fundamental rights includes the Central Government, Parliament, State Governments, State Legislatures, and all local authorities. This definition determines who is bound by fundamental rights obligations."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_13",
        "title": "Laws Inconsistent with Fundamental Rights",
        "text": "Article 13 — Laws inconsistent with or in derogation of the fundamental rights: (1) All laws in force in the territory of India immediately before the commencement of this Constitution, in so far as they are inconsistent with the provisions of this Part, shall, to the extent of such inconsistency, be void. (2) The State shall not make any law which takes away or abridges the rights conferred by this Part and any law made in contravention of this clause shall, to the extent of the contravention, be void.\n\nArticle 13 in Simple Words\nAny law that violates fundamental rights is void. The State cannot make laws that take away fundamental rights. This establishes the doctrine of judicial review in India."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_16",
        "title": "Equality of Opportunity in Public Employment",
        "text": "Article 16 — Equality of opportunity in matters of public employment: (1) There shall be equality of opportunity for all citizens in matters relating to employment or appointment to any office under the State. (2) No citizen shall, on grounds only of religion, race, caste, sex, descent, place of birth, residence or any of them, be ineligible for, or discriminated against in respect of, any employment or office under the State. (4) Nothing in this article shall prevent the State from making any provision for the reservation of appointments or posts in favour of any backward class of citizens.\n\nArticle 16 in Simple Words\nAll citizens have equal opportunity in government employment. No discrimination based on religion, race, caste, sex, etc. However, the State can provide reservations for backward classes."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_39A",
        "title": "Equal Justice and Free Legal Aid",
        "text": "Article 39A — Equal justice and free legal aid: The State shall secure that the operation of the legal system promotes justice, on a basis of equal opportunity, and shall, in particular, provide free legal aid, by suitable legislation or schemes or in any other way, to ensure that opportunities for securing justice are not denied to any citizen by reason of economic or other disabilities.\n\nArticle 39A in Simple Words\nThe State must ensure the legal system provides equal justice and free legal aid to those who cannot afford it. No citizen should be denied justice due to economic disabilities."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_243",
        "title": "Panchayati Raj — Local Self Government",
        "text": "Article 243 — Definitions relating to Panchayats: In this Part, unless the context otherwise requires, 'district' means a district in a State; 'Gram Sabha' means a body consisting of persons registered in the electoral rolls relating to a village comprised within the area of Panchayat at the village level; 'intermediate level' means a level between the village and district levels specified by the Governor of a State by public notification. (Part IX added by the Constitution (Seventy-third Amendment) Act, 1992.)\n\nArticle 243 in Simple Words\nThis article defines key terms for the three-tier Panchayati Raj system of local self-governance at village, intermediate, and district levels, added by the 73rd Constitutional Amendment in 1992."
    },
    {
        "law": "CONSTITUTION",
        "section": "Article_370",
        "title": "Special Provisions for Jammu and Kashmir (Abrogated)",
        "text": "Article 370 — Temporary provisions with respect to the State of Jammu and Kashmir: This article granted special autonomous status to Jammu and Kashmir. It was effectively abrogated on 5 August 2019 by a Presidential Order (C.O. 272) under Article 367, followed by the Jammu and Kashmir Reorganisation Act, 2019. The Supreme Court in December 2023 upheld the abrogation as constitutionally valid.\n\nArticle 370 in Simple Words\nFormerly granted special status to Jammu and Kashmir (own constitution, separate laws, autonomy). Abrogated in August 2019 by the Modi government. The Supreme Court upheld this decision in December 2023."
    },
]

def main():
    statutes_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "statutes", "statutes.jsonl")
    statutes_path = os.path.abspath(statutes_path)

    # Check if articles already exist
    existing_articles = set()
    if os.path.exists(statutes_path):
        with open(statutes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("law") == "CONSTITUTION":
                    existing_articles.add(entry["section"])

    added = 0
    skipped = 0
    with open(statutes_path, "a", encoding="utf-8") as f:
        for article in ARTICLES:
            if article["section"] in existing_articles:
                print(f"  SKIP (already exists): {article['section']}")
                skipped += 1
                continue
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
            added += 1
            print(f"  ADDED: {article['section']} — {article['title']}")

    print(f"\nDone! Added {added} articles, skipped {skipped}.")
    print(f"Total articles available: {len(ARTICLES)}")

if __name__ == "__main__":
    main()
