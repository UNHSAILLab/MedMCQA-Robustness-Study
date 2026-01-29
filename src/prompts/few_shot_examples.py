"""Curated few-shot examples for medical QA tasks.

Examples are selected to:
1. Cover different medical subjects
2. Demonstrate reasoning patterns
3. Have unambiguous correct answers
"""

MEDMCQA_FEW_SHOT_EXAMPLES = [
    {
        "question": "Which of the following is NOT a feature of Cushing syndrome?",
        "options": {
            "A": "Moon facies",
            "B": "Central obesity",
            "C": "Hypotension",
            "D": "Purple striae"
        },
        "correct_answer": "C",
        "reasoning": (
            "Moon facies (A) is a classic feature due to fat redistribution. "
            "Central obesity (B) occurs from cortisol causing truncal fat deposition. "
            "Purple striae (D) result from skin thinning and visible blood vessels. "
            "However, Cushing syndrome causes HYPERTENSION, not hypotension (C), "
            "due to mineralocorticoid effects of excess cortisol."
        ),
        "subject": "Medicine"
    },
    {
        "question": "The normal anion gap is:",
        "options": {
            "A": "8-12 mEq/L",
            "B": "16-20 mEq/L",
            "C": "20-24 mEq/L",
            "D": "24-28 mEq/L"
        },
        "correct_answer": "A",
        "reasoning": (
            "The anion gap is calculated as Na+ - (Cl- + HCO3-). "
            "The normal range is 8-12 mEq/L. "
            "An elevated anion gap (>12) suggests metabolic acidosis from unmeasured anions "
            "like lactate, ketoacids, or toxins."
        ),
        "subject": "Biochemistry"
    },
    {
        "question": "Which muscle is NOT a rotator cuff muscle?",
        "options": {
            "A": "Supraspinatus",
            "B": "Infraspinatus",
            "C": "Deltoid",
            "D": "Subscapularis"
        },
        "correct_answer": "C",
        "reasoning": (
            "The rotator cuff consists of four muscles: Supraspinatus (A), "
            "Infraspinatus (B), Teres minor, and Subscapularis (D). "
            "These muscles stabilize the glenohumeral joint. "
            "The Deltoid (C) is NOT part of the rotator cuff - it's a superficial "
            "shoulder muscle responsible for arm abduction."
        ),
        "subject": "Anatomy"
    },
    {
        "question": "First-line treatment for H. pylori infection includes all EXCEPT:",
        "options": {
            "A": "Proton pump inhibitor",
            "B": "Clarithromycin",
            "C": "Metronidazole",
            "D": "Amphotericin B"
        },
        "correct_answer": "D",
        "reasoning": (
            "Standard triple therapy for H. pylori includes a PPI (A), "
            "clarithromycin (B), and either amoxicillin or metronidazole (C). "
            "Amphotericin B (D) is an antifungal agent, not used for bacterial "
            "H. pylori infection."
        ),
        "subject": "Pharmacology"
    },
    {
        "question": "Reed-Sternberg cells are pathognomonic of:",
        "options": {
            "A": "Non-Hodgkin lymphoma",
            "B": "Hodgkin lymphoma",
            "C": "Multiple myeloma",
            "D": "Chronic lymphocytic leukemia"
        },
        "correct_answer": "B",
        "reasoning": (
            "Reed-Sternberg cells are large, binucleated cells with prominent nucleoli "
            "giving an 'owl-eye' appearance. They are pathognomonic (specifically "
            "diagnostic) for Hodgkin lymphoma (B). Non-Hodgkin lymphoma (A) lacks "
            "these cells. Multiple myeloma shows plasma cells, and CLL shows "
            "small mature lymphocytes."
        ),
        "subject": "Pathology"
    }
]


PUBMEDQA_FEW_SHOT_EXAMPLES = [
    {
        "question": "Does vitamin D supplementation reduce the risk of cardiovascular events?",
        "context": (
            "[BACKGROUND] Observational studies have suggested that low vitamin D levels "
            "are associated with increased cardiovascular risk. [METHODS] We conducted a "
            "meta-analysis of 21 randomized controlled trials examining vitamin D "
            "supplementation and cardiovascular outcomes. [RESULTS] The pooled analysis "
            "showed no significant reduction in major cardiovascular events (RR 0.98, "
            "95% CI 0.93-1.04) with vitamin D supplementation compared to placebo."
        ),
        "answer": "no",
        "reasoning": (
            "The meta-analysis of 21 RCTs found no significant reduction in cardiovascular "
            "events with vitamin D supplementation (RR 0.98, CI includes 1.0). "
            "Despite observational associations, the randomized evidence does not support "
            "a protective effect."
        )
    },
    {
        "question": "Is metformin associated with reduced cancer incidence in diabetic patients?",
        "context": (
            "[BACKGROUND] Metformin has shown anti-proliferative effects in preclinical "
            "studies. [METHODS] We analyzed data from 47,351 diabetic patients, comparing "
            "cancer incidence in metformin users vs non-users over 5 years. [RESULTS] "
            "Metformin users had significantly lower cancer incidence (HR 0.72, 95% CI "
            "0.64-0.81, p<0.001) after adjusting for age, BMI, and smoking status."
        ),
        "answer": "yes",
        "reasoning": (
            "The study found a significant 28% reduction in cancer incidence among "
            "metformin users (HR 0.72, CI 0.64-0.81, p<0.001). The confidence interval "
            "is entirely below 1.0, indicating a robust protective association."
        )
    },
    {
        "question": "Can machine learning models predict sepsis onset with high accuracy?",
        "context": (
            "[BACKGROUND] Early sepsis detection is crucial for improving patient outcomes. "
            "[METHODS] We developed and validated an ML model using vital signs and lab "
            "values from 52,000 ICU admissions. [RESULTS] The model achieved AUROC of 0.85 "
            "but with modest positive predictive value (PPV 0.42) at the optimal threshold, "
            "raising concerns about clinical utility due to high false positive rates."
        ),
        "answer": "maybe",
        "reasoning": (
            "The model shows good discrimination (AUROC 0.85) but the modest PPV of 0.42 "
            "means many false alarms. Whether this constitutes 'high accuracy' depends on "
            "clinical context - the trade-off between early detection and alarm fatigue "
            "remains uncertain."
        )
    }
]
