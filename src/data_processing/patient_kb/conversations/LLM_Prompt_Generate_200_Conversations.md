# **PROMPT FOR LLM: Generate High-Quality Patient-Clinician Bowel Prep Conversations**

## **YOUR TASK**

Generate high-quality, natural patient-clinician conversations about colonoscopy bowel preparation. These are dialogues between patients and knowledgeable bowel prep nurses (NOT chatbot interactions). They will be used as training examples for an AI-powered patient education chatbot.

**Generate:** 200 unique conversations (or specify your desired number)

**Output format:** Excel/CSV with these columns:
- conversation_id (1, 2, etc.)
- turn_number (1, 2, 3, etc.)
- risk_tier (Low, Medium, or High)
- prep_type (Prepopik, Suprep, Moviprep, or PEG Electrolytes GoLYTELY)
- appointment_time (e.g., "8:00 AM", "10:30 AM", "2:00 PM")
- patient_message
- clinician_response
- escalated_to_physician (Yes or No)
- escalation_reason (text or blank)

---

## **CRITICAL CONTEXT**

### **Who is Speaking?**

**Clinician = Bowel prep nurse** (expert in prep protocols, NOT a chatbot)
- Can answer routine questions about diet, timing, medications per protocol
- Escalates to physician when: beyond nursing scope, requires medical decision-making, safety concern
- When escalating: **Tells patient to contact their specialist** (does NOT say "we'll contact them for you")

**Patient = Someone scheduled for colonoscopy**
- Ages 35-78, various backgrounds
- Communication styles: anxious, practical, frustrated, straightforward, nervous
- Medical backgrounds: healthy to complex (diabetes, heart disease, kidney disease, prior failed preps)

### **Escalation Means:** Nurse tells patient to contact physician/specialist themselves

**NOT:** Chatbot escalating to human  
**NOT:** Nurse saying "I'll have our team contact you"

---

## **CONVERSATION DISTRIBUTION**

### **Risk Tiers (must match these percentages):**
- **Low Risk:** 50% of conversations (100 out of 200)
- **Medium Risk:** 30% of conversations (60 out of 200)  
- **High Risk:** 20% of conversations (40 out of 200)

### **Conversation Length (vary naturally):**
- Short (2-3 turns): ~35%
- Medium (4-5 turns): ~40%
- Long (6-7 turns): ~25%

---

## **RISK TIER DEFINITIONS & TONE REQUIREMENTS**

### **LOW RISK (50% - Conversations 1-100)**

**Patient Profile:**
- Healthy adults, minimal comorbidities
- No prior failed prep
- No high-risk medications
- First-time or routine screening

**REQUIRED CLINICIAN TONE:**
- ✅ **Friendly openings:** 40%+ should start with "Great question!", "Good thinking!", "Totally understandable!", "Perfect!", "I know!"
- ✅ **Reassuring:** "You're doing great!", "This is normal", "Many people ask this"
- ✅ **Conversational:** Use contractions (you'll, it's, can't), casual phrasing
- ✅ **Encouraging:** "You'll do fine!", exclamation points for enthusiasm
- ❌ **Avoid:** "Critical", "You must", "Essential" - too harsh for Low risk

**Example Low-Risk Conversation:**
```
Patient: "can i have gatorade or does it have to be plain water?"
Nurse: "Great question! Yes, Gatorade is perfectly fine as long as it's not red or purple. Clear or yellow Gatorade works well and helps with electrolytes too. Many people find it easier to drink than plain water!"
```

### **MEDIUM RISK (30% - Conversations 101-160)**

**Patient Profile:**
- Controlled comorbidities (diabetes, hypertension, obesity)
- Some medication complexity
- No prior failed prep
- Age-related concerns (60+)

**REQUIRED CLINICIAN TONE:**
- ✅ **Warm but professional:** No "Great question!" but still friendly
- ✅ **Emphasizes importance:** "This is important", "Make sure to", "Be sure to"
- ✅ **Provides context:** "Given your medical history...", "With your conditions..."
- ✅ **More detailed:** Longer explanations, more specifics
- ✅ **Some urgency:** Can use "important", "should", "need to"

**Example Medium-Risk Conversation:**
```
Patient: "i have hypertension do i take my blood pressure pill in the morning?"
Nurse: "Take your blood pressure medication in the morning with a small sip of water. This is important to keep your BP controlled during the procedure. Most cardiac and BP meds should be continued."
```

### **HIGH RISK (20% - Conversations 161-200)**

**Patient Profile:**
- Multiple/severe comorbidities
- Prior inadequate bowel prep (2+ times)
- High-risk medications (anticoagulants, GLP-1 agonists)
- Advanced kidney disease, cirrhosis, heart failure

**REQUIRED CLINICIAN TONE:**
- ✅ **Directive and urgent:** 30%+ should use "Critical", "You must", "Essential", "Immediately"
- ✅ **Explicit consequences:** "This could affect your safety", "Given your risk factors"
- ✅ **Action-oriented:** Clear directives, specific instructions
- ✅ **Longer responses:** 20%+ longer than Low-risk responses
- ✅ **Safety-focused:** Minimal reassurance, focus on instructions
- ✅ **Escalates frequently:** See escalation rules below

**Example High-Risk Conversation:**
```
Patient: "i failed prep twice before what should i do differently"
Nurse: "Given your history of inadequate preparations, you need a physician-designed enhanced protocol. This requires earlier diet modifications, higher volume prep, and close monitoring. You need to contact your gastroenterologist to have them create a specialized prep plan for your situation. The standard prep likely won't be sufficient given your history."
[ESCALATED: Yes - High risk patient: Prior inadequate prep requiring custom protocol]
```

---

## **ESCALATION RULES (CRITICAL)**

### **NEVER Escalate (nurse can handle):**
- Clear liquid diet questions
- Prep timing for split-dose
- Taste improvement tips
- Mild nausea management
- Transportation/escort questions
- Post-procedure recovery questions
- General blood pressure medications (continue with sip of water)
- Thyroid medications (continue)
- Cholesterol medications (continue)
- Simple diabetes questions (hold oral meds day-of, can drink sugary clear liquids)

### **ALWAYS Escalate (100% of the time):**

1. **Blood thinners/Anticoagulants:**
   - Keywords: warfarin, Coumadin, Eliquis, Xarelto, Plavix, aspirin (for procedure), Pradaxa, apixaban, rivaroxaban, clopidogrel
   - Escalation reason: "Anticoagulant management"
   - Response template: "Blood thinner management requires coordination with your prescribing physician. The timing depends on your specific indication and risk factors. **Please contact your cardiologist [or whoever prescribed it] to get specific instructions on when to hold and restart it.**"

2. **GLP-1 Agonists:**
   - Keywords: Ozempic, Wegovy, Mounjaro, Trulicity, Victoza, semaglutide, tirzepatide
   - Escalation reason: "GLP-1 medication timing"
   - Response template: "GLP-1 medications like [drug name] require careful timing decisions from your prescribing provider. The stop time varies by medication and your situation. **Please contact your endocrinologist or prescribing physician to ask them when to hold the [drug name] before your procedure.** They'll give you the safest timing based on your dosing schedule."

3. **Prior Inadequate Prep (High-Risk patients):**
   - If patient mentions: failed prep, cancelled before, prep didn't work, inadequate prep
   - Escalation reason: "High risk patient: Prior inadequate prep requiring custom protocol"
   - Response template: "Given your history of inadequate preparations, you need a physician-designed enhanced protocol. This requires earlier diet modifications, higher volume prep, and close monitoring. **You need to contact your gastroenterologist to have them create a specialized prep plan for your situation.** The standard prep likely won't be sufficient given your history."

4. **Advanced Kidney Disease:**
   - Keywords: CKD stage 4, CKD stage 5, dialysis, kidney failure, renal failure
   - Escalation reason: "Advanced kidney disease requiring specialist clearance"
   - Response template: "With advanced kidney disease, your nephrologist must approve the specific prep before you proceed. Many standard preps can be dangerous with reduced kidney function due to electrolyte imbalances. **Please contact your kidney doctor to confirm which prep is safe for your level of function.** Do not proceed until you've confirmed with them."

5. **Cirrhosis/Advanced Liver Disease:**
   - Keywords: cirrhosis, liver failure, advanced liver disease
   - Escalation reason: "Cirrhosis requiring specialist input"
   - Response template: "With cirrhosis, you need condition-specific guidance from your hepatologist or gastroenterologist. Some prep considerations apply with liver disease. **Please contact your liver specialist to confirm the prep plan is appropriate for your liver condition.** If you haven't received specific instructions, reach out to them before proceeding."

6. **Complex Insulin Management:**
   - Keywords: insulin pump, multiple daily injections with complex regimen
   - Escalation reason: "Complex insulin management"
   - Response template: "With your insulin regimen, you need specific guidance from your endocrinologist or diabetes provider. Insulin dosing during prep varies based on your specific regimen and control. **Please contact your diabetes care provider to get a clear plan for insulin adjustments during the prep day.** This is important to manage safely."


### **Escalation Rate Targets:**
- Low Risk: 0-3% (rare - only if emergency scenario)
- Medium Risk: 3-7% (occasional anticoagulant/complex cases)
- High Risk: 40-60% (frequent specialist coordination needed)

### **KEY PRINCIPLE FOR ALL ESCALATIONS:**

**The nurse tells the patient to contact the specialist themselves.**

The nurse does NOT say:
- ❌ "I'll have our team contact you"
- ❌ "We'll reach out to your doctor"
- ❌ "I'm flagging this for our team to call you"
- ❌ "Our gastroenterology team will contact you"

The nurse DOES say:
- ✅ "Please contact your [specialist]"
- ✅ "You need to reach out to your [doctor]"
- ✅ "Call your [specialist] to get specific guidance"
- ✅ "Contact your prescribing physician"

---

## **GENERIC CONTACT PHRASES (CRITICAL - NO SPECIFIC NAMES/NUMBERS)**

### **✅ ALWAYS USE (Generic):**
- "Your cardiologist"
- "Your endocrinologist"
- "Your nephrologist"
- "Your prescribing physician"
- "Your primary care provider"
- "The gastroenterology team"
- "Your gastroenterologist"
- "Your diabetes care provider"
- "Your liver specialist"
- "The endoscopy unit"
- "Emergency services"

### **❌ NEVER USE (Specific):**
- Phone numbers: "XXX-XXX-XXXX", "555-1234"
- Hospital names: "Mayo Clinic", "Rochester campus"
- Doctor names: "Dr. Smith", "Dr. Johnson"
- Facility names: "St. Mary's Hospital"
- "Call 911" → Say "Call emergency services"

---

## **NATURAL LANGUAGE REQUIREMENTS**

### **Patient Message Style:**

**✅ DO:**
- Use lowercase (not proper capitalization)
- Include casual language: "gonna", "wanna", "kinda"
- Use conversation starters for turn 2+: "okay", "so", "thanks", "got it", "one more thing", "also", "wait"
- Make typos occasionally (but readable): "recieve" → "receive"
- Ask follow-up questions naturally
- Vary question phrasing extensively

**❌ DON'T:**
- Make patient messages formal or clinical
- Use perfect grammar and capitalization
- Include specific personal details (test results, specific doses, dates, doctor names)

**Good Patient Examples:**
- "what does clear liquids actually mean?"
- "okay got it. one more thing can i have gatorade?"
- "i take metformin for diabetes. do i skip that?"
- "everyone says the prep tastes awful. any tips?"
- "so my appointment is at 10am when do i take the second dose?"

**Bad Patient Examples:**
- "Please advise regarding my medication regimen." ❌ Too formal
- "My A1C is 8.2, is that acceptable?" ❌ Specific test result
- "Dr. Johnson told me to take 20 units of insulin." ❌ Specific doctor/dose
- "I had my colonoscopy at Mayo Clinic on March 15, 2024." ❌ Specific facility/date

### **Clinician Response Style:**

**✅ DO:**
- Match tone to risk tier (see above)
- Use natural, conversational language
- Provide evidence-based medical information
- Cite general principles, not specific studies
- Escalate appropriately using generic phrases
- Tell patient to contact their specialist (not "we'll contact them")

**❌ DON'T:**
- Include phone numbers or facility names
- Mention specific doctors by name
- Use medical jargon without explanation
- Give contradictory medical advice
- Say "I'll have our team contact you" when escalating

---

## **COMORBIDITY-SPECIFIC QUESTIONS (IMPORTANT)**

### **Design Principle:**
Questions must be **generalizable** - applicable to ALL patients with that condition, not one specific patient.

### **✅ GOOD (Generalizable):**

**Diabetes:**
- "i have diabetes what do i do about my medications during prep?"
- "i'm diabetic and take metformin should i skip it?"
- "diabetic here worried about blood sugar with no food all day"
- "i take insulin how do i dose during bowel prep?"

**Kidney Disease:**
- "i have chronic kidney disease which prep is safe?"
- "i'm on dialysis can i do bowel prep?"
- "kidney function is low what prep should i avoid?"

**Heart Disease:**
- "i have heart failure is the prep safe for me?"
- "i have a pacemaker any precautions?"
- "i've had a heart attack can i do bowel prep?"

**Prior GI History:**
- "i've had failed prep before what can i do differently?"
- "my last colonoscopy was cancelled for bad prep help?"
- "i have chronic constipation will prep work?"
- "i've had part of my colon removed any modifications?"

### **❌ BAD (Too Specific - Avoid):**
- "my a1c is 8.2 is that okay?" ❌ Specific test result
- "dr johnson told me 20 units insulin" ❌ Specific doctor, dose
- "my creatinine is 2.8 can i take prep?" ❌ Specific lab value
- "i had surgery at mayo in march 2023" ❌ Specific facility, date

---

## **QUESTION CATEGORIES TO COVER (30 Categories)**

Ensure variety across all 200 conversations. Each category should appear multiple times with different phrasings:

### **General Prep Questions:**
1. Clear liquid diet definition
2. Solid food restrictions
3. Split-dose timing
4. Prep taste improvement
5. Nausea/vomiting management
6. Brown stool morning-of
7. One week diet modifications
8. Prep storage and mixing
9. What to bring to procedure

### **Medication Questions:**
10. General medication management
11. Blood pressure medications
12. Blood thinners (ESCALATE)
13. Diabetes medications (oral)
14. Insulin management
15. GLP-1 agonists (ESCALATE)
16. Cholesterol medications
17. Thyroid medications

### **Comorbidity-Specific:**
18. Diabetes concerns
19. Kidney disease safety
20. Heart failure/cardiac
21. Prior inadequate prep (High-risk: ESCALATE)
22. Chronic constipation
23. Prior colon surgery
24. Gastroparesis
25. Hypertension
26. Obesity concerns

### **Logistics:**
27. Transportation/escort
28. Post-procedure recovery
29. Return to work timing
30. When to resume eating

---

## **CONVERSATION FLOW EXAMPLES**

### **Example 1: Low-Risk, 4-Turn Natural Flow**

**CONV000023** | Low Risk | Suprep | 10:00 AM

**Turn 1:**
Patient: "so my procedure is at 10am when exactly should i start the second dose?"
Nurse: "Good planning ahead! For your 10 AM appointment, you'll take the first half between 5-9 PM the night before. Then start the second half at 5 AM and finish by 6 AM. Stop all liquids by 6 AM."
Escalated: No

**Turn 2:**
Patient: "okay got it. and can i have coffee in the morning before i come in?"
Nurse: "Yes, black coffee is fine but you need to stop ALL liquids 4 hours before. So nothing after 6 AM, including coffee. You can have it earlier in the morning though!"
Escalated: No

**Turn 3:**
Patient: "perfect. what if i start feeling nauseous while drinking the prep?"
Nurse: "Nausea can happen during prep. If it does, take a short break for 10-15 minutes until you feel better, then resume at a slower pace. You're doing fine!"
Escalated: No

**Turn 4:**
Patient: "thanks that helps. one more thing can i have gatorade or just water?"
Nurse: "Great question! Gatorade is perfectly fine as long as it's not red or purple. Many people find it easier to drink than plain water. You're doing great with your prep planning!"
Escalated: No

---

### **Example 2: Medium-Risk, 3-Turn Diabetes**

**CONV000127** | Medium Risk | Prepopik | 8:30 AM

**Turn 1:**
Patient: "i have type 2 diabetes and take metformin and glipizide what do i do about those the day of?"
Nurse: "For diabetes medications: hold oral diabetes pills on the day of your procedure. So skip both the metformin and glipizide that morning. Monitor your blood sugar as usual and have glucose tablets available. If you haven't received specific instructions from your diabetes provider, please confirm this plan with them."
Escalated: No

**Turn 2:**
Patient: "what if my blood sugar goes too low during prep when im not eating?"
Nurse: "Good thinking. You can drink clear liquids with sugar - apple juice, regular (not diet) Jell-O, or regular soda. These count as clear liquids AND help maintain blood sugar. Check your sugar regularly and treat lows as you normally would."
Escalated: No

**Turn 3:**
Patient: "okay that makes sense. and i have high blood pressure too do i take those pills?"
Nurse: "Yes, take your blood pressure medications in the morning with a small sip of water. This is important to keep your BP controlled during the procedure. Most cardiac and BP meds should be continued."
Escalated: No

---

### **Example 3: High-Risk, 2-Turn Anticoagulant**

**CONV000178** | High Risk | Moviprep | 11:00 AM

**Turn 1:**
Patient: "i take eliquis twice a day for blood clots when do i stop that before the test?"
Nurse: "Blood thinner management requires coordination with your prescribing physician. The timing for holding and restarting Eliquis depends on your indication and bleeding risk. Please contact your cardiologist or prescribing provider to get specific instructions on when to stop and restart it."
Escalated: Yes
Escalation Reason: Anticoagulant management

**Turn 2:**
Patient: "okay ill call my cardiologist. do i take my other medications normally?"
Nurse: "Yes, continue your other medications as usual unless your doctor specifically told you otherwise. Take them with a small sip of water the morning of your procedure. Your cardiologist will give you specific guidance on the Eliquis timing."
Escalated: No

---

### **Example 4: High-Risk, 3-Turn Prior Failed Prep**

**CONV000185** | High Risk | PEG Electrolytes | 7:30 AM

**Turn 1:**
Patient: "i failed prep twice before and im really worried this is going to happen again"
Nurse: "Given your history of inadequate preparations, you need a physician-designed enhanced protocol. This requires earlier diet modifications, higher volume prep, and close monitoring. You need to contact your gastroenterologist to have them create a specialized prep plan for your situation. The standard prep likely won't be sufficient given your history."
Escalated: Yes
Escalation Reason: High risk patient: Prior inadequate prep requiring custom protocol

**Turn 2:**
Patient: "what does that mean exactly what will be different?"
Nurse: "The physician will likely recommend starting a low-residue diet 2-3 days early instead of just the day before, possibly higher volume prep or additional agents, and specific monitoring. When you contact them, they'll design a protocol tailored to your history to maximize success this time."
Escalated: No

**Turn 3:**
Patient: "okay. and i also have chronic constipation does that matter?"
Nurse: "Yes, that's another factor for them to consider in your customized plan. Chronic constipation is one reason preps can fail, so the enhanced protocol will account for slower bowel transit. Make sure to mention this when you speak with your gastroenterologist."
Escalated: No

---

### **Example 5: High-Risk, 2-Turn GLP-1 Medication**

**CONV000192** | High Risk | Suprep | 10:00 AM

**Turn 1:**
Patient: "i take ozempic for my diabetes when should i stop that before the test?"
Nurse: "GLP-1 medications like Ozempic require careful timing decisions from your prescribing provider. The stop time varies by medication and your specific situation. Please contact your endocrinologist or prescribing physician to ask them when to hold the Ozempic before your procedure. They'll give you the safest timing based on your dosing schedule."
Escalated: Yes
Escalation Reason: GLP-1 medication timing

**Turn 2:**
Patient: "why is that different from my other diabetes medications?"
Nurse: "GLP-1 medications delay gastric emptying, which can affect sedation safety. Your endocrinologist needs to advise on timing. When you contact them, mention you're having a procedure with sedation so they can give you the right guidance."
Escalated: No

---

### **Example 6: High-Risk, 2-Turn Advanced Kidney Disease**

**CONV000197** | High Risk | Prepopik | 9:00 AM

**Turn 1:**
Patient: "i have stage 4 kidney disease which prep should i avoid?"
Nurse: "With advanced kidney disease, your nephrologist must approve the specific prep before you proceed. Many standard preps can be dangerous with stage 4 CKD due to electrolyte imbalances. Please contact your kidney doctor to confirm which prep is safe for your level of function. Don't start the prep until you've gotten their approval."
Escalated: Yes
Escalation Reason: High risk patient: Advanced kidney disease requiring specialist clearance

**Turn 2:**
Patient: "what could happen if i take the wrong prep?"
Nurse: "With reduced kidney function, certain preps can cause dangerous electrolyte imbalances that could affect your heart rhythm and overall safety. This is why your nephrologist needs to approve it. Make sure you confirm with them before proceeding."
Escalated: No

---

## **QUALITY CHECKLIST**

Before generating each conversation, ensure:

### **Content Quality:**
- [ ] Clinically accurate medical information and prep details following the document attached
- [ ] Risk-appropriate tone (friendly for Low, professional for Medium, directive for High)
- [ ] Natural patient language (lowercase, casual, conversational)
- [ ] Proper escalation logic (ALWAYS for anticoagulants, GLP-1s, prior failed prep)
- [ ] No specific names, numbers, facilities, or doctors
- [ ] Generic contact phrases only
- [ ] Escalations tell patient to contact specialist (NOT "we'll contact them")
- [ ] The patient does not know their risk level

### **Uniqueness:**
- [ ] Each patient message is phrased differently from others on same topic
- [ ] Each clinican message is phrased differently from others on same topic
- [ ] Persona comes through in communication style
- [ ] Natural conversation flow (not Q&A robot)
- [ ] Follow-up questions connect to previous answers

### **Variety:**
- [ ] Different conversation lengths (2-7 turns)
- [ ] All 30 question categories represented
- [ ] Mix of simple and complex patient situations
- [ ] Different appointment times and prep types

---

## **OUTPUT INSTRUCTIONS**

1. **Generate exactly 200 conversations**
2. **Follow distribution:**
   - Conversations 1-100: Low Risk
   - Conversations 101-160: Medium Risk
   - Conversations 161-200: High Risk

3. **Number conversations sequentially:** CONV000001, CONV000002, ... CONV000200

4. **Create Excel/CSV with columns:**
   - conversation_id
   - turn_number
   - risk_tier
   - prep_type
   - appointment_time
   - patient_message
   - clinician_response
   - escalated_to_physician
   - escalation_reason

5. **Validate before finishing:**
   - Risk distribution matches (100 Low, 60 Medium, 40 High)
   - Escalation rates: Low 0-3%, Medium 3-7%, High 40-60%
   - Low-risk friendly tone: 40%+ start with friendly phrases
   - High-risk directive tone: 30%+ use "Critical", "Must", "Essential"
   - 100% unique patient messages
   - All anticoagulant and GLP-1 questions escalate
   - All escalations tell patient to contact specialist (not "we'll contact them")

---

## **BEGIN GENERATION**

Start generating 200 high-quality conversations following all rules above. Focus on creating natural, persona-driven dialogues that sound like real phone calls between patients and expert bowel prep nurses.

Remember: When escalating, the nurse tells the patient to contact their specialist, NOT "I'll have our team contact you."

---
