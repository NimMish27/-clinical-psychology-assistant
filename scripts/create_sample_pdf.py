"""Create a sample clinical psychology PDF for testing the pipeline."""
from pathlib import Path
import fitz

output_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

doc = fitz.open()
doc.new_page()

content = """
Cognitive Behavioral Therapy for Depression

Cognitive Behavioral Therapy (CBT) is a structured, time-limited psychotherapy
that has been extensively researched and validated for the treatment of major
depressive disorder. CBT is based on the cognitive model of depression, which
posits that distorted thinking patterns and maladaptive behaviors contribute to
the maintenance of depressive symptoms.

The core principles of CBT include:
1. Cognitive restructuring - identifying and challenging negative automatic thoughts
2. Behavioral activation - increasing engagement in rewarding activities
3. Problem-solving therapy - developing practical solutions to life stressors
4. Relapse prevention - building skills to maintain treatment gains

Research evidence demonstrates that CBT is as effective as antidepressant
medication for mild to moderate depression, and combination treatment (CBT plus
medication) is superior to either treatment alone for moderate to severe depression.

Diagnostic Criteria for Major Depressive Disorder (DSM-5)

According to the DSM-5, a diagnosis of Major Depressive Disorder requires five
or more of the following symptoms to have been present during the same two-week
period, representing a change from previous functioning:
- Depressed mood most of the day, nearly every day
- Markedly diminished interest or pleasure in all, or almost all, activities
- Significant weight loss or weight gain, or decrease or increase in appetite
- Insomnia or hypersomnia nearly every day
- Psychomotor agitation or retardation nearly every day
- Fatigue or loss of energy nearly every day
- Feelings of worthlessness or excessive guilt
- Diminished ability to think or concentrate
- Recurrent thoughts of death or suicidal ideation

At least one of the symptoms must be either depressed mood or loss of interest
or pleasure.

Anxiety Disorders: Treatment Approaches

Generalized Anxiety Disorder (GAD) is characterized by excessive, uncontrollable
worry about a variety of topics. CBT for GAD typically includes:
- Psychoeducation about the nature of anxiety
- Relaxation training and mindfulness techniques
- Cognitive restructuring of worry-related thoughts
- Exposure therapy for worry triggers
- Time management and problem-solving skills

Panic Disorder involves recurrent unexpected panic attacks and fear of future
attacks. Effective treatment components include:
- Interoceptive exposure to physical sensations
- Cognitive restructuring of catastrophic misinterpretations
- Breathing retraining and relaxation techniques
- Situational exposure to avoided situations

Evidence-Based Practice in Clinical Psychology

The American Psychological Association defines evidence-based practice as the
integration of the best available research with clinical expertise in the context
of patient characteristics, culture, and preferences. Key empirically supported
treatments include:
- CBT for depression, anxiety disorders, and eating disorders
- Dialectical Behavior Therapy (DBT) for borderline personality disorder
- Prolonged Exposure therapy for PTSD
- Interpersonal Therapy for depression
- Mindfulness-Based Cognitive Therapy for relapse prevention in depression

Clinical assessment should include structured diagnostic interviews, validated
self-report measures, and behavioral observations. Regular monitoring of
treatment progress using standardized outcome measures is essential for
evidence-based practice.
""".strip()

page = doc[0]
page.insert_text(fitz.Point(50, 50), content, fontsize=9)

pdf_path = output_dir / "cbt_depression_manual.pdf"
doc.save(str(pdf_path))
doc.close()
print(f"Created sample PDF: {pdf_path}")
