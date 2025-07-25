from pydantic import BaseModel, Field
import typing as t

class Sentiment(BaseModel):
    chain_of_thought: str
    sentiment: t.Literal["Hawkish", "Dovish", "Neutral"]

SENTIMENT_PROMPT = """
        You are an expert in macroeconomics and central bank policy.
    You are given a set of statements from a FED press conference transcript.
    Your task is to analyze this set of phrases and determine whether the FED is hawkish, dovish or neutral.
    Here are the criteria for your analysis:
        Level 1: Strongly Hawkish
        - Emphasis on Inflation Control: Clear priority on fighting inflation, even at the expense of growth or employment.
        - Aggressive Rate Hike Signals: Explicit suggestions of rapid or large interest rate increases (e.g., “We will continue raising rates aggressively until inflation is under control”).
        - Tightening Commitment: Use of firm language like “We must act decisively,” “We are committed to tightening,” with minimal hedging.
        - De-emphasis on Growth/Employment: Statements downplay economic growth or labor market concerns in favor of price stability.
        - Tone: Assertive and uncompromising, with strong, decisive language. 
          Example: "We will not hesitate to take necessary actions to control inflation."
        
        Level 2: Moderately Hawkish
        - Inflation as a Primary Concern: Inflation control is prioritized, but with recognition of economic risks.
        - Gradual Tightening Signals: Indications of continued rate hikes but at a measured pace (e.g., “We expect further rate increases in the coming months”).
        - Balanced Commitment: Phrases like “We are prepared to adjust rates as needed” show intent to tighten but with flexibility.
        - Acknowledgment of Risks: Mentions potential impact on growth but frames it as necessary to control inflation.
        - Tone: Confident but measured, with a balance of assertiveness and caution.
          Example: "While inflation remains a concern, we will proceed with caution to ensure economic stability."
        
        Level 3: Moderately Dovish
        - Focus on Economic Growth: Greater emphasis on supporting economic growth and employment.
        - Pause or Slowdown in Tightening: References to slowing or pausing rate hikes (e.g., “We need to assess the impact of prior increases”).
        - Conditional Softening: Statements like “We will be data-dependent moving forward” imply openness to less aggressive policy.
        - Recognition of Downside Risks: Concern over potential recession or market volatility is acknowledged.
        - Tone: Cautious and flexible, with language that suggests openness to change.
          Example: "We are closely monitoring economic indicators and will adjust our policies as necessary."
        
        Level 4: Strongly Dovish
        - Prioritizing Growth Over Inflation: Clear focus on supporting growth and employment, even if inflation persists.
        - Rate Cuts or Policy Easing Signals: Direct or indirect hints at rate cuts or easing (e.g., “We are prepared to provide additional support if necessary”).
        - Soft Language on Inflation: Downplaying inflation risks (e.g., “Inflation is expected to moderate on its own”).
        - Commitment to Accommodative Policy: Firm statements about maintaining accommodative monetary policy (e.g., “Rates will remain low for an extended period”).
        - Tone: Reassuring and supportive, with language that emphasizes stability and support.
          Example: "Our priority is to support economic growth and ensure a stable recovery."

    Please output your analysis and the result in form of a number between 1 and 4. You MUST Finish your answer with "Result: " and put result into hash tags like this: ##result##.

    Your answer:
"""

class EmploymentInflationLevels(BaseModel):
    chain_of_thought: str
    employment_level: t.Literal["Strong", "Weak", "Balanced"]
    inflation_level: t.Literal["High", "Moderate", "Low"]

    def extract_results(self):
        return {
            'employment_level': self.employment_level,
            'inflation_level': self.inflation_level
        }

EMPLOYMENT_INFLATION_LEVELS_PROMPT = """
    You are an international expert in macroeconomics and central bank policy.

    Your task is to analyze the provided extracted statements regarding the labor market and inflation and classify their current **state** as described by the Federal Reserve.

    **Thinking Process:**
    1. Identify statements where the FED describes **the current condition** of the labor market and inflation.
    2. Determine if the labor market is characterized as **strong, weak, or balanced** based on indicators such as employment growth, unemployment levels, and wage trends.
    3. Determine if inflation is characterized as **high, moderate, or low** based on price levels, inflation expectations, and cost pressures.
    4. Prioritize **explicit** statements that directly describe current conditions rather than speculative or forward-looking comments.

    **Classification Rules:**
    - **Labor Market Condition:**
      - **Classify as "Strong"** if statements indicate:
        - Strong job growth
        - Historically low unemployment
        - Rising wages
        - Tight labor market
      - **Classify as "Weak"** if statements indicate:
        - Weak or declining job growth
        - Rising unemployment
        - Signs of labor market deterioration
        - Concerns about layoffs or slow wage growth
      - **Classify as "Balanced"** if statements indicate:
        - A labor market that is stable but not particularly strong or weak.
        - A mix of positive and negative signals that do not suggest a clear trend.
        - No signs of overheating or significant weakness.

    - **Inflation Condition:**
      - **Classify as "High"** if statements indicate:
        - Inflation remains persistently elevated.
        - Concerns about sustained price pressures.
        - Risk of unanchored inflation expectations.
        - Price levels are increasing across multiple categories.
      - **Classify as "Moderate"** if statements indicate:
        - Inflation is stable and not significantly increasing or decreasing.
        - No strong concerns about excessive inflation or deflation.
        - A mix of factors where inflation is neither overheating nor rapidly falling.
      - **Classify as "Low"** if statements indicate:
        - Inflation is subdued or declining.
        - Price pressures are easing.
        - Supply-side improvements have reduced inflation risk.

    **Final Output Format:**
       Labor Market Condition: [Strong/Weak/Balanced]  
       Inflation Condition: [High/Moderate/Low]

    Provide your answer based solely on the provided text, ensuring logical consistency in classification.
"""

class EmploymentInflationDynamics(BaseModel):
    chain_of_thought: str
    employment_dynamics: t.Literal["Improving", "Deteriorating", "Stable"]
    inflation_dynamics: t.Literal["Rising", "Falling", "Stable"]

    def extract_results(self):
        return {
            'employment_dynamics': self.employment_dynamics,
            'inflation_dynamics': self.inflation_dynamics
        }
    
EMPLOYMENT_INFLATION_DYNAMICS_PROMPT = """
        You are an international expert in macroeconomics and central bank policy.

    Your task is to analyze the provided extracted statements regarding the labor market and inflation, and classify their **recent dynamics** as described by the Federal Reserve.

    **Thinking Process:**
    1. Identify statements where the FED discusses **changes, risks, or trends** in employment and inflation.
    2. Determine whether the labor market and inflation are **improving, deteriorating, or stable** based on recent indicators and FED commentary.
    3. Look for explicit mentions of **downside risks, uncertainties, or acceleration trends** that indicate future direction.
    4. If statements contain **mixed signals**, prioritize **explicit references** over vague or indirect mentions.

    **Classification Rules:**
    - **Labor Market Dynamics:**
      - **Classify as "Improving"** if:
        - Job growth is accelerating.
        - Unemployment is decreasing.
        - Wages are rising at a faster pace.
      - **Classify as "Deteriorating"** if:
        - Job growth is slowing or turning negative.
        - Unemployment is rising.
        - Concerns about layoffs or wage stagnation are increasing.
      - **Classify as "Stable"** if:
        - No significant changes in employment conditions.
        - The labor market remains steady with no major positive or negative shifts.

    - **Inflation Dynamics:**
      - **Classify as "Rising"** if:
        - Inflationary pressures are increasing.
        - Concerns about accelerating price growth.
        - Mentions of supply constraints or cost-push factors driving inflation higher.
      - **Classify as "Falling"** if:
        - Inflation is decelerating.
        - Price pressures are easing.
        - Improvements in supply chains or cost reductions are mentioned.
      - **Classify as "Stable"** if:
        - No significant inflationary changes.
        - Inflation remains at current levels without notable shifts.

    **Final Output Format:**
       Labor Market Dynamics: [Improving/Deteriorating/Stable]  
       Inflation Dynamics: [Rising/Falling/Stable]

    Provide your answer based solely on the provided text, ensuring logical consistency in classification.
"""

class EmploymentInflationConcern(BaseModel):
    chain_of_thought: str = Field(description="Your reasoning before making your decision")
    main_concern: t.Literal["Employment", "Inflation", "Balanced"] = Field(description="The main concern for the FED in this report")

    def extract_results(self):
        return {
            'main_concern': self.main_concern
        }

EMPLOYMENT_INFLATION_CONCERN_PROMPT = """
    You are an international expert in macroeconomics and central bank policy.

    Your task is to analyze the provided extracted statements regarding employment and inflation, and determine whether the FED's primary concern is employment, inflation, or if it maintains a balanced stance between the two.

    **Thinking Process:**
    1. Identify all statements discussing employment and inflation.
    2. Assess the relative emphasis on each:
       - Look for **strong language** indicating urgency (e.g., "our priority," "primary concern," "critical issue").
       - Count the number of mentions for employment vs. inflation.
       - Determine if one topic is explicitly tied to FED policy decisions more than the other.
    3. Compare the tone:
       - Does one issue appear as a **more urgent risk** (e.g., inflation is "unacceptably high," or employment "is deteriorating fast")?
       - Does the FED suggest a strong **policy action** for one issue while treating the other as secondary?
    4. If the focus appears **balanced**, check for the following:
       - Does the FED mention both employment and inflation equally, without prioritizing one?  
       - If neither issue seems dominant, classify as **Balanced Stance** instead of forcing one choice.  

    **Classification Rules:**
    - **Choose "Employment is the main concern"** if:
      - The FED **primarily discusses** employment risks, job market conditions, or wage concerns.
      - Employment conditions are explicitly linked to policy decisions.
      - Inflation is mentioned but appears less urgent than labor market risks.

    - **Choose "Inflation is the main concern"** if:
      - The FED **primarily discusses** inflation risks, price stability, or inflation expectations.
      - Inflation conditions are explicitly linked to policy decisions.
      - Employment is mentioned but appears less urgent than inflation control.

    - **Choose "Balanced stance"** if:
      - The FED **gives equal weight** to both inflation and employment concerns.
      - Statements indicate a **dual focus** without prioritizing one issue over the other.
      - No clear preference is apparent in policy decisions or urgency.

    **Final Output Format:**
       Main Concern: [Employment / Inflation / Balanced]

    Provide your answer based solely on the provided text, ensuring logical consistency in classification.
"""

class InterestRateTrajectory(BaseModel):
    chain_of_thought: str
    interest_rate_trajectory: t.Literal["Raise", "Cut", "Hold"]

    def extract_results(self):
        return {
            'interest_rate_trajectory': self.interest_rate_trajectory
        }

INTEREST_RATE_TRAJECTORY_PROMPT = """
    You are an international expert in macroeconomics and central bank policy.

    Your task is to analyze the provided extracted statements and determine the **main trajectory** of the Federal Reserve’s interest rate policy. Specifically, classify whether the FED is planning to:
    - **Raise rates**
    - **Cut rates**
    - **Hold rates at the current level**

    **Thinking Process:**
    1. **Identify direct and indirect statements about rate trajectory**  
       - Look for **explicit** references to rate hikes, cuts, or holding steady.
       - Identify **implicit** cues where the FED describes conditions consistent with future rate changes (without directly stating an intention).
    
    2. **Determine if the FED is leaning toward raising, cutting, or holding rates**  
       - Are they discussing the need for **further policy firming, restrictive policy, or maintaining higher rates**?  
         → **This suggests a plan to RAISE rates.**  
       - Are they mentioning **accommodative policy, loosening monetary conditions, or providing support to the economy**?  
         → **This suggests a plan to CUT rates.**  
       - Are they emphasizing **a wait-and-see approach, stability, or watching data before making a move**?  
         → **This suggests they intend to HOLD rates at the current level.**
    
    3. **Weigh the overall message**  
       - If statements contain mixed signals, determine which stance is **more dominant** based on:  
         - The strength of language (e.g., "we will continue tightening" vs. "we might consider cuts")  
         - The number of times a stance is referenced  
         - The degree of certainty (strong commitment vs. vague hints)

    4. **Ignore discussions about economic risks or conditions that justify decisions**  
       - Your task is **only to determine what the FED is planning to do**, not why they are doing it.  
       - Do not consider discussions about inflation, employment, or economic stability unless they contain **a direct reference to rate trajectory**.

    **Classification Rules:**
    - **Choose "Raise Rates"** if:
      - The FED explicitly states its intent to **increase interest rates**.
      - Phrases such as **"further tightening," "additional rate hikes," "policy firming is needed,"** appear.
      - There is a clear emphasis on restrictive monetary policy.

    - **Choose "Cut Rates"** if:
      - The FED explicitly signals an intention to **lower interest rates**.
      - Statements reference **"policy accommodation," "rate cuts," "easing financial conditions,"** or similar phrases.
      - Language suggests a shift toward supporting economic activity.

    - **Choose "Hold Rates"** if:
      - The FED expresses **no clear intention** to raise or cut rates.
      - Statements indicate a **wait-and-see** stance, **data dependency, or a preference for maintaining current policy**.
      - Phrases like **"holding rates steady," "keeping policy unchanged," "waiting for more data,"** are used.

    **Final Output Format:**
       Interest Rate Trajectory: [Raise / Cut / Hold]

    Provide your answer based solely on the provided text, ensuring logical consistency in classification.
"""

class BalanceSheetTrajectory(BaseModel):
    chain_of_thought: str
    balance_sheet_trajectory: t.Literal["Expand", "Reduce", "Unchanged"]

    def extract_results(self):
        return {
            'balance_sheet_trajectory': self.balance_sheet_trajectory
        }

BALANCE_SHEET_TRAJECTORY_PROMPT = """
    You are an international expert in macroeconomics and central bank policy.

    Your task is to analyze the provided extracted statements and determine the **main trajectory** of the Federal Reserve’s balance sheet policy, specifically its approach to asset purchases or quantitative tightening (QT). Classify the FED’s stance into one of the following categories:
    - **Increase asset purchases** (Expanding the balance sheet, QE)
    - **Decrease asset purchases** (Reducing the balance sheet, QT)
    - **Keep asset purchases unchanged** (Maintaining the current policy)

    **Thinking Process:**
    1. **Identify direct and indirect statements about asset purchases**  
       - Look for **explicit** references to quantitative easing (QE), quantitative tightening (QT), balance sheet reduction, or asset purchase programs.  
       - Identify **implicit** cues where the FED describes conditions that suggest future balance sheet expansion or contraction without directly stating an intention.

    2. **Determine if the FED is signaling an increase, decrease, or maintaining asset purchases**  
       - Are they discussing the need to **expand liquidity, provide additional market support, or increase holdings of Treasuries/MBS**?  
         → **This suggests a plan to INCREASE asset purchases.**  
       - Are they referencing **reducing the balance sheet, winding down asset purchases, or allowing assets to roll off without reinvestment**?  
         → **This suggests a plan to DECREASE asset purchases.**  
       - Are they emphasizing **stability, maintaining current levels, or waiting for further market developments**?  
         → **This suggests they intend to KEEP asset purchases UNCHANGED.**

    3. **Weigh the overall message**  
       - If statements contain mixed signals, determine which stance is **more dominant** based on:  
         - The strength of language (e.g., "we will continue reducing holdings" vs. "we might consider adjustments")  
         - The number of times a stance is referenced  
         - The degree of certainty (strong commitment vs. vague hints)

    4. **Ignore discussions about economic risks or conditions that justify decisions**  
       - Your task is **only to determine what the FED is planning to do**, not why they are doing it.  
       - Do not consider discussions about inflation, employment, or financial stability unless they contain **a direct reference to asset purchases or balance sheet trajectory**.

    **Classification Rules:**
    - **Choose "Increase asset purchases"** if:
      - The FED explicitly states an intention to **expand its balance sheet or purchase more assets**.
      - Phrases such as **"additional liquidity support," "resuming asset purchases," "expanding quantitative easing,"** appear.
      - Language suggests an accommodative policy shift.

    - **Choose "Decrease asset purchases"** if:
      - The FED explicitly signals an intention to **reduce its balance sheet holdings**.
      - Statements reference **"reducing holdings," "winding down purchases," "allowing assets to mature without reinvestment,"** or similar phrases.
      - Language suggests a restrictive or tightening policy stance.

    - **Choose "Keep asset purchases unchanged"** if:
      - The FED expresses **no clear intention** to expand or reduce asset purchases.
      - Statements indicate a **wait-and-see** stance, **data dependency, or maintaining the current balance sheet policy**.
      - Phrases like **"maintaining current holdings," "continuing at the current pace," "keeping policy unchanged,"** are used.

    **Final Output Format:**
       Asset Purchase Trajectory: [Expand / Reduce / Unchanged]

    Provide your answer based solely on the provided text, ensuring logical consistency in classification.
"""

class Guidance(BaseModel):
    chain_of_thought: str
    guidance: t.Literal["Clear Guidance", "Conditional Guidance", "No Guidance"]

    def extract_results(self):
        return {
            'guidance': self.guidance
        }

GUIDANCE_PROMPT = """
    You are an international expert in macroeconomics and central bank policy.

    Your task is to analyze the provided extracted statements regarding the Federal Reserve’s future interest rate policy and classify the **explicitness and commitment** of its guidance into one of three categories:

    - **Clear Guidance on the Next Step**
    - **Conditional Guidance**
    - **No Guidance Given**

    **Thinking Process:**
    1. **Identify statements related to future interest rate decisions.**  
       - Focus only on direct references to the FED’s next policy move.
       - Ignore general economic commentary unless it is explicitly linked to rate policy.

    2. **Determine the Type of Guidance Given:**  
       - If the FED **clearly states its next action**, classify as **Clear Guidance on the Next Step**.
       - If the FED **provides a possible action but ties it to specific conditions**, classify as **Conditional Guidance**.
       - If the FED **avoids providing a directional view and insists on full data dependence**, classify as **No Guidance Given**.

    3. **Classification Rules:**  
       - **Choose "Clear Guidance on the Next Step"** if:
         - The FED makes an **unambiguous statement** about the next move.
         - Even if exact timing or numbers are missing, the **direction of policy is clear**.
         - Example: *"We believe additional tightening will be necessary to achieve price stability."*
         - Example: *"We will raise rates at our next meeting."*
       - **Choose "Conditional Guidance"** if:
         - The FED suggests a **possible** action but makes it **dependent on economic conditions**.
         - Must include a **clear directional bias** (e.g., rate hikes or cuts).
         - Example: *"If inflation does not slow, we will need to raise rates further."*
         - Example: *"Further rate hikes could be appropriate if inflation does not show sufficient progress."*
       - **Choose "No Guidance Given"** if:
         - The FED refuses to indicate any future direction and **relies fully on incoming data**.
         - No **clear** bias toward tightening, easing, or holding rates.
         - Example: *"We will assess economic conditions on a meeting-by-meeting basis."*
         - Example: *"We are not making any predictions about our next steps."*

    **Example Analysis:**
    - **Statement:** *"If inflation does not show continued progress, we may need to raise rates further. However, we are also mindful of the lagged effects of monetary policy and will assess all economic data before making any decisions."*
    - **Classification:** Conditional Guidance  
    - **Reasoning:** The statement provides a **possible action** (rate hikes), but it is **contingent** on inflation progress and overall economic data.

    **Final Output Format:**
       Rate Policy Guidance: [Clear Guidance on the Next Step / Conditional Guidance / No Guidance Given]

    Provide your answer based solely on the provided text, ensuring logical consistency in classification.
"""


