import typing as t
import os

from pydantic import BaseModel, Field

SENTIMENT_RANKING_PROMPT = """
    
    You are given a set of statements from a FED press conference transcript that are helping to understand whether the FED provided hawkish or dovish sentiment.
    You are also given a set of statements from another FED press conference.
    Your task is to determine which of the two press conferences is more hawkish or dovish.
    Think carefully before answering. Output your reasoning and the result ranking as 1 if the first document is more hawkish (means that the second document is more dovish) or -1 if the second document is more hawkish (means that the first document is more dovish).

    Press Conference 1:
    {statements1}

    Press Conference 2:
    {statements2}

"""

class EmploymentLevelRankingResponse(BaseModel):
    set_a_employment_levels: str = Field(description="The perceived condition of the labor market in Set A")
    set_a_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the labor market in Set A")
    set_b_employment_levels: str = Field(description="The perceived condition of the labor market in Set B")
    set_b_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the labor market in Set B")
    ranking_reasoning: str = Field(description="The reasoning for the ranking result between Set A and Set B")
    result: t.Literal[1, -1, 0] = Field(description="The ranking result. 1 if Set B is stronger, -1 if Set B is weaker, 0 if they are the same.")


EMPLOYMENT_LEVEL_RANKING_PROMPT = """
    
    You are analyzing Federal Reserve statements about the labor market across two periods. Your task is to determine whether the perceived condition of the labor market in Set B (later statement) is stronger, weaker, or approximately the same compared to Set A (earlier statement).
    Evaluate labour market trend on three directions:
    1. Trend in Economic Indicators (Implied or Explicit)
        Summarize whether Set B describes:
        - Job creation: Stronger, weaker, or unchanged?
        - Unemployment rate: Rising, falling, or stable?
        - Wage growth: Accelerating, decelerating, or unchanged?
        - Labor force participation: Expanding, shrinking, or stable?
        - Job openings and labor demand: Increasing, decreasing, or unchanged?

        How to analyze:
        - If Set B indicates a broad improvement in these indicators relative to Set A, the labor market is likely stronger.
        - If Set B suggests worsening conditions in these areas, the labor market is likely weaker.
        - If conditions are similar or changes are minor, the labor market is likely the same.

    2. Change in Sentiment and Tone
        Compare how the Federal Reserve describes the labor market:

        - Identify if stronger or more confident language is used in Set B (e.g., "resilient," "robust") compared to Set A.
        - Identify if weaker or more cautious language appears in Set B (e.g., "softening," "slowing").
        - If language remains neutral and unchanged, the perception is likely the same.

        How to analyze:
        - If the tone in Set B is more optimistic or confident, the labor market is likely stronger.
        - If the tone in Set B is more cautious or negative, the labor market is likely weaker.
        - If the tone is similar, the labor market is likely the same.

    3. Final Classification
        After evaluating these two dimensions, classify the labor market condition into one of the following categories:

        - Stronger Labor Market (The FED describes a more optimistic or confident labor market.)
        - Weaker Labor Market (The FED describes a more cautious or negative labor market.)
        - Labor Market Condition Unchanged (The tone and language used in both statements is similar.)

    First analyze the labor market condition and sentiment in Set A and then in Set B. After that based on that think about the final classification.
"""

class EmploymentDynamicsRankingResponse(BaseModel):
    set_a_employment_trend: str = Field(description="The perceived employment trend of the labor market in Set A")
    set_a_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the labor market in Set A")
    set_b_employment_trend: str = Field(description="The perceived employment trend of the labor market in Set B")
    set_b_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the labor market in Set B")
    ranking_reasoning: str = Field(description="The reasoning for the ranking result between Set A and Set B")
    result: t.Literal[1, -1, 0] = Field(description="The ranking result. 1 if Set B trend is acceleration, -1 if Set B trend is deceleration, 0 if trends are stable across both sets.")

EMPLOYMENT_DYNAMICS_RANKING_PROMPT = """
    
    You are analyzing Federal Reserve statements about the labor market across two periods.
    Your task is to compare the narrative on employment momentum between two Federal Reserve press conferences and determine whether the perceived trajectory of employment conditions in Set B (later statement) shows acceleration, stabilization, or deceleration compared to Set A (earlier statement).

    Assessment Criteria:
    Analyze the change in momentum across the following three aspects:

    1. Shift in the Description of Employment Trends
    Compare how the FED describes employment trends in Set A vs. Set B:

    - If employment was previously improving but is now described as stabilizing or weakening, this indicates a deceleration.
    - If employment was previously weakening but is now stabilizing or improving, this indicates an acceleration.
    - If employment dynamics were described similarly in both statements (e.g., both describe continued improvement, cooling, or stability), then momentum is unchanged.
        Key factors to analyze:

        - Job creation (Has the FED shifted from "strong" to "moderate" job gains? Or vice versa?)
        - Unemployment rate trends (Is the FED more concerned about rising unemployment compared to the previous statement?)
        - Labor demand & job openings (Is demand for workers increasing or softening at a different rate than before?)
        - Wage growth (Is wage growth accelerating, steady, or slowing differently than in Set A?)
    2. Change in Sentiment and Tone Regarding Momentum
        Compare the FED's language around employment momentum:
        - If Set B introduces a more cautious or uncertain tone compared to Set A, it signals a deceleration in employment momentum.
        - If Set B introduces a more confident or stronger tone about employment conditions, it signals an acceleration in employment momentum.
        - If the tone remains consistent, it suggests momentum is unchanged.
        Key shifts to look for:
        - Stronger to neutral language (e.g., "robust job gains" → "job gains remain solid" = Deceleration)
        - Neutral to stronger language (e.g., "moderating labor market" → "hiring picking up" = Acceleration)
        - Consistently neutral/strong/cautious language (No momentum shift)
    3. Change in Policy Implications Related to Employment
        Analyze whether the FED's stance on employment-related risks and policy actions has changed:

        - If the FED was concerned about overheating labor markets in Set A but signals more balance in Set B, this suggests a deceleration in employment momentum.
        - If the FED was previously concerned about employment cooling but is now more optimistic about hiring and job growth, this suggests an acceleration in employment momentum.
        - If the FED's employment-related policy stance remains similar, momentum is unchanged.

    4. Final Classification
        After evaluating these three dimensions, classify the change in employment momentum into one of the following categories:

        - Acceleration in Employment Momentum (Employment trend is strengthening compared to the previous press conference.)
            - The FED describes employment as improving at a faster rate or reversing a prior slowdown.
        - Stable Employment Momentum (No major shift in employment trajectory.)
            - The FED continues to describe employment with a similar level of optimism or caution.
        - Deceleration in Employment Momentum (Employment trend is slowing down compared to the previous press conference.)
            - The FED indicates that job gains are stabilizing or slowing, or previously strong labor conditions are cooling.
"""

class InflationLevelRankingResponse(BaseModel):
    set_a_inflation_levels: str = Field(description="The perceived inflation levels of the inflation in Set A")
    set_a_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the inflation in Set A")
    set_b_inflation_levels: str = Field(description="The perceived inflation levels of the inflation in Set B")
    set_b_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the inflation in Set B")
    ranking_reasoning: str = Field(description="The reasoning for the ranking result between Set A and Set B")
    result: t.Literal[1, -1, 0] = Field(description="The ranking result. 1 if Set B is higher, -1 if Set B is lower, 0 if they are the same.")

INFLATION_LEVEL_RANKING_PROMPT = """
     You are analyzing Federal Reserve statements about inflation levels across two periods. 
    Your task is to determine whether the perceived condition of inflation in Set B (later statement) is higher, lower, or approximately the same compared to Set A (earlier statement).

    Analyze the difference in inflation levels between Set A and Set B using the following 3 aspects:
    1. Trend in Inflation Indicators (Implied or Explicit)
        Summarize whether Set B describes the following inflation-related indicators as higher, lower, or unchanged compared to Set A:

        - Overall inflation rate: Rising, falling, or stable?
        - Core inflation (excluding food & energy): Increasing, decreasing, or unchanged?
        - Goods vs. services inflation: Is inflation rising or falling more in goods or services?
        - Wage growth and cost pressures: Are labor costs contributing more or less to inflation?
        - Inflation expectations (consumer & business): Are inflation expectations rising, falling, or stable?

        How to analyze:
        - If Set B describes broad price increases or persistent inflation pressures compared to Set A, inflation levels are likely higher.
        - If Set B indicates falling inflation, moderating price growth, or easing pressures, inflation levels are likely lower.
        - If inflation trends remain similar or changes are minor, inflation levels are likely the same.

    2. Change in Sentiment and Tone Regarding Inflation
        Compare how the Federal Reserve describes inflation in Set A vs. Set B:

        - Identify if stronger or more concerned language is used in Set B (e.g., "persistent inflationary pressures," "inflation remains elevated") compared to Set A.
        - Identify if less concerned or more positive language appears in Set B (e.g., "inflationary pressures are easing," "price growth is moderating").
        - If language remains neutral and unchanged, inflation levels are likely the same.

        How to analyze:
        - If the tone in Set B is more concerned or alarmed about inflation than Set A, inflation levels are likely higher.
        - If the tone in Set B is more confident or relaxed, inflation levels are likely lower.
        - If the tone is similar, inflation levels are likely the same.

    3. Final Classification
        After evaluating these three dimensions, classify inflation levels into one of the following categories:

        - Higher Inflation Levels (Inflation is perceived as more elevated than before.)
            - The FED describes stronger price pressures or increased inflation concerns.
        - Stable Inflation Levels (No major shift in inflation levels.)
            - The FED continues to describe inflation at similar levels.
        - Lower Inflation Levels (Inflation is perceived as easing.)
            - The FED signals that price pressures are moderating and inflation is declining.
"""

class InflationDynamicsRankingResponse(BaseModel):
    set_a_inflation_dynamics: str = Field(description="The perceived inflation dynamics of the inflation in Set A")
    set_a_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the inflation in Set A")
    set_b_inflation_dynamics: str = Field(description="The perceived inflation dynamics of the inflation in Set B")
    set_b_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the inflation in Set B")
    ranking_reasoning: str = Field(description="The reasoning for the ranking result between Set A and Set B")
    result: t.Literal[1, -1, 0] = Field(description="The ranking result. 1 if Set B is reaccelerating, -1 if Set B is decelerating, 0 if inflation dynamics are stable across both sets.")

INFLATION_DYNAMICS_RANKING_PROMPT = """
    
    Your task is to analyze how the inflation momentum has evolved between two Federal Reserve press conferences. 
    Specifically, determine whether the perceived trajectory of inflation in Set B (later statement) shows faster inflationary pressures, stabilization, or a stronger disinflation trend compared to Set A (earlier statement).

    Analyze the change in inflation momentum based on the following directions:
    
    1. Shift in the Description of Inflation Dynamics
    Compare how the FED describes inflation's trajectory in Set A vs. Set B:

        - If inflation was previously slowing but is now stabilizing or rising again, this indicates a reacceleration in inflation momentum.
        - If inflation was previously rising but is now moderating or stabilizing, this indicates a deceleration in inflation momentum.
        - If inflation momentum was described similarly in both statements (e.g., inflation continued accelerating or cooling at a similar pace), then momentum is unchanged.

        Key Factors to Analyze:
        - Overall Inflation Trends: Is inflation accelerating, decelerating, or stabilizing differently than before?
        - Core vs. Headline Inflation: Is core inflation (excluding food & energy) showing a different pace of change compared to the prior statement?
        - Sectoral Inflation Shifts: Are price increases spreading to new areas, or are pressures easing across more sectors?
        - Wage-Price Relationship: Has the FED's assessment of wage-driven inflation changed?
        - Inflation Expectations: Have business or consumer expectations for inflation adjusted at a different rate?

        How to analyze:
        - If Set B suggests inflation is moderating more quickly than before, inflation momentum is decelerating.
        - If Set B suggests inflation was easing but is now stabilizing or re-accelerating, inflation momentum is accelerating.
        - If Set B describes a similar pace of inflation trends as Set A, inflation momentum is unchanged.

    2. Change in Sentiment and Tone Regarding Inflation Momentum
        Compare the FED's language on inflation trajectory:
        - If Set B introduces a stronger sense of urgency or concern compared to Set A (e.g., "inflation remains stubborn," "risks of reacceleration"), this suggests inflation momentum is accelerating.
        - If Set B is more optimistic about inflation trends than Set A (e.g., "inflation continues to cool," "downward trend is becoming broad-based"), this suggests inflation momentum is decelerating.
        - If the tone remains consistent, inflation momentum is unchanged.

        Key Shifts to Look For:
        - Strong to neutral language (e.g., "inflation is falling rapidly" → "inflation is stabilizing" = Acceleration)
        - Neutral to strong language (e.g., "inflation remains elevated" → "inflation is easing faster than expected" = Deceleration)
        - Consistently neutral/cautious/optimistic language (No momentum shift)

    Final Classification
        After evaluating these three dimensions, classify the change in inflation momentum into one of the following categories:
        - Reaccelerating Inflation Momentum (Inflationary pressures are intensifying compared to the previous press conference.)
            - The FED describes inflation as increasing at a faster pace or slowing at a weaker rate than before.
        - Stable Inflation Momentum (No major shift in inflation trajectory.)
            - The FED describes inflation momentum at a similar pace as before, whether rising or falling.
        - Decelerating Inflation Momentum (Inflationary pressures are easing at a faster pace than before.)
            - The FED signals that inflation is moderating more rapidly or broadening in its decline.
"""

class InterestRateProjectionRankingResponse(BaseModel):
    set_a_interest_rate_projection: str = Field(description="The perceived interest rate projection of the interest rate in Set A")
    set_a_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the interest rate in Set A")
    set_a_implications_and_market_signals: str = Field(description="The perceived implications and market signals of the interest rate in Set A")
    set_b_interest_rate_projection: str = Field(description="The perceived interest rate projection of the interest rate in Set B")
    set_b_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the interest rate in Set B")
    set_b_implications_and_market_signals: str = Field(description="The perceived implications and market signals of the interest rate in Set B")
    ranking_reasoning: str = Field(description="The reasoning for the ranking result between Set A and Set B")
    result: t.Literal[1, -1, 0] = Field(description="The ranking result. 1 if Set B is more accommodative, -1 if Set B is more restrictive, 0 if they are the same.")

INTEREST_RATE_PROJECTION_RANKING_PROMPT = """
 
Your task is to analyze Federal Reserve statements on interest rate policy across two periods and determine whether the guidance in Set B (later statement) suggests a shift toward a more accommodative (easing) stance, a more restrictive (tightening) stance, or remains unchanged compared to Set A (earlier statement).

Evaluate the change in interest rate projections based on the following three aspects:
    1. Shift in Forward Guidance on Interest Rates
        Compare how the FED's outlook for interest rate policy has evolved between Set A and Set B.

        - If Set B suggests a softer stance than Set A (e.g., signaling potential rate cuts, pausing hikes sooner than expected, or focusing on downside risks), this suggests a shift toward a more accommodative/easing policy.
        - If Set B suggests a firmer stance than Set A (e.g., emphasizing the need for higher rates for longer, delaying rate cuts, or stressing upside risks to inflation), this suggests a shift toward a more restrictive/tightening policy.
        - If Set B maintains a similar policy outlook to Set A, interest rate guidance is likely unchanged.

        Key Factors to Analyze:

        - Expected rate trajectory: Is the FED adjusting its expected path of rate changes (hikes, pauses, or cuts)?
        - Conditions for rate adjustments: Has the FED shifted its criteria for tightening or easing (e.g., labor market conditions, inflation persistence)?
        - Reference to policy lags: Is the FED emphasizing the delayed effects of previous hikes more in Set B than Set A?
        - Balance of risks: Has the FED shifted its assessment of inflation vs. growth risks that would impact rate decisions?

    2. Change in Sentiment and Tone Regarding Policy Stance
        Compare the tone and language used in Set A vs. Set B regarding interest rate policy:

        - If Set B expresses greater confidence that inflation is under control or signals a willingness to ease policy, this suggests a shift toward a more accommodative stance.
        - If Set B expresses greater concern about inflation risks or the need for tighter policy, this suggests a shift toward a more restrictive stance.
        - If the tone remains neutral and unchanged, it suggests no meaningful shift in policy stance.

        Key Shifts to Look For:

        - Tighter to neutral language (e.g., "additional tightening may be needed" → "monetary policy is sufficiently restrictive" = Easing shift)
        - Neutral to tighter language (e.g., "policy is appropriate" → "further tightening may be warranted" = Tightening shift)
        - Consistently neutral/cautious/optimistic language (No policy shift)

    3. Change in Policy Implications and Market Signals
        Determine whether the FED's guidance on future policy actions has changed:

        - If Set B introduces a stronger case for potential rate cuts or a pause, this suggests a shift toward easing.
        - If Set B reinforces the need for additional hikes or prolonged higher rates, this suggests a shift toward tightening.
        - If Set B suggests the same overall approach as Set A, interest rate guidance remains unchanged.

        Key Policy Signals to Compare:

        - Explicit mentions of rate cuts or hikes: Does Set B suggest rate cuts sooner than Set A?
        - Guidance on "higher for longer": Has Set B softened or reinforced this language?
        - Reference to economic conditions: Does Set B introduce new risks that justify policy shifts?

    Final Classification
        After evaluating these three dimensions, classify the change in interest rate guidance into one of the following categories:

        - More Accommodative Policy Shift (The FED signals greater willingness to ease monetary policy.)
            - The FED indicates rate cuts, pauses hikes earlier, or expresses confidence that restrictive policy is no longer needed.
        - Stable Policy Stance (No major shift in interest rate guidance.)
            - The FED maintains a similar rate outlook, with no meaningful changes in guidance.
        - More Restrictive Policy Shift (The FED signals a need for tighter or prolonged restrictive policy.)
            - The FED suggests further rate hikes, delays cuts, or expresses concern about inflation risks.
"""

class BalanceSheetProjectionRankingResponse(BaseModel):
    set_a_balance_sheet_projection: str = Field(description="The perceived balance sheet projection of the balance sheet in Set A")
    set_a_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the balance sheet in Set A")
    set_a_implications_and_market_signals: str = Field(description="The perceived implications and market signals of the balance sheet in Set A")
    set_b_balance_sheet_projection: str = Field(description="The perceived balance sheet projection of the balance sheet in Set B")
    set_b_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the balance sheet in Set B")
    set_b_implications_and_market_signals: str = Field(description="The perceived implications and market signals of the balance sheet in Set B")
    ranking_reasoning: str = Field(description="The reasoning for the ranking result between Set A and Set B")
    result: t.Literal[1, -1, 0] = Field(description="The ranking result. 1 if Set B is more expansionary, -1 if Set B is more contractionary, 0 if balance sheet policy is stable across both sets.")

BALANCE_SHEET_PROJECTION_RANKING_PROMPT = """
Your task is to analyze Federal Reserve statements on balance sheet policy across two periods and determine whether the guidance in Set B (later statement) suggests a shift toward a more accommodative (easing) stance, a more restrictive (tightening) stance, or remains unchanged compared to Set A (earlier statement).

Evaluate the change in balance sheet trajectory based on the following three aspects:

1. Shift in Forward Guidance on Balance Sheet Policy
    Compare how the FED's outlook for balance sheet adjustments has evolved between Set A and Set B.
        - If Set B suggests a more accommodative stance than Set A (e.g., slowing the pace of balance sheet reduction, stopping Quantitative Tightening (QT), or signaling potential asset purchases), this suggests a shift toward a more expansionary balance sheet policy.
        - If Set B suggests a more restrictive stance than Set A (e.g., accelerating QT, reducing holdings at a faster pace, or maintaining reductions longer than expected), this suggests a shift toward a more contractionary balance sheet policy.
        - If Set B maintains a similar balance sheet outlook to Set A, the balance sheet policy is likely unchanged.

    Key Factors to Analyze:
        - Pace of Balance Sheet Reduction (QT): Is the FED adjusting the speed of reducing its holdings of Treasuries and mortgage-backed securities (MBS)?
        - Conditions for Slowing or Stopping QT: Has the FED changed the criteria for pausing QT (e.g., concerns over liquidity or financial stability)?
        - Potential for Balance Sheet Expansion (QE): Does Set B introduce the possibility of renewed asset purchases?
        - Market Liquidity Considerations: Is the FED expressing new concerns about liquidity that could impact balance sheet policy?

2. Change in Sentiment and Tone Regarding Balance Sheet Policy
    Compare the tone and language used in Set A vs. Set B regarding balance sheet strategy:
        - If Set B expresses greater concern about tightening financial conditions or liquidity risks, this suggests a shift toward a more accommodative balance sheet policy.
        - If Set B expresses greater confidence in QT or stresses the need for continued balance sheet reductions, this suggests a shift toward a more restrictive balance sheet policy.
        - If the tone remains neutral and unchanged, it suggests no meaningful shift in balance sheet policy.

    Key Shifts to Look For:
        - Tighter to neutral language (e.g., "QT will continue as planned" → "We will monitor liquidity conditions before adjusting QT" = Easing shift)
        - Neutral to tighter language (e.g., "We are carefully reducing assets" → "QT will proceed at an appropriate pace for longer" = Tightening shift)
        - Consistently neutral/cautious/optimistic language (No policy shift)  

3. Change in Policy Implications and Market Signals
    Determine whether the FED's guidance on future balance sheet actions has changed:
        - If Set B introduces a stronger case for slowing QT, halting reductions, or considering balance sheet expansion, this suggests a shift toward easing.
        - If Set B reinforces the need for further balance sheet reduction or delays discussions of balance sheet expansion, this suggests a shift toward tightening.
        - If Set B suggests the same overall approach as Set A, balance sheet guidance remains unchanged.

        Key Policy Signals to Compare:
        - Explicit mentions of QT adjustments: Does Set B suggest slowing, stopping, or reversing QT sooner than Set A?
        - Liquidity concerns: Is Set B more focused on financial stability or reserve adequacy than Set A?
        - Market signaling: Are there new hints of balance sheet policy shifts to prevent market disruptions?

After evaluating these three dimensions, classify the change in balance sheet guidance into one of the following categories:

        - More Expansionary Balance Sheet Policy (The FED signals greater willingness to slow or halt QT, or consider balance sheet expansion.)
            - The FED indicates a slower pace of asset runoff, concerns over liquidity, or potential asset purchases.
        - Stable Balance Sheet Policy (No major shift in balance sheet trajectory.)
            - The FED maintains a similar approach to asset reduction or expansion as before.
        - More Contractionary Balance Sheet Policy (The FED signals a need for further balance sheet reductions or prolonged QT.)
            - The FED suggests accelerating QT, maintaining a restrictive balance sheet stance, or delaying future balance sheet expansion.
"""

class GuidanceRankingResponse(BaseModel):
    set_a_guidance: str = Field(description="The perceived guidance of the guidance in Set A")
    set_a_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the guidance in Set A")
    set_b_guidance: str = Field(description="The perceived guidance of the guidance in Set B")
    set_b_sentiment_and_tone: str = Field(description="The perceived sentiment and tone of the guidance in Set B")
    ranking_reasoning: str = Field(description="The reasoning for the ranking result between Set A and Set B")
    result: t.Literal[1, -1, 0] = Field(description="The ranking result. 1 if Set B is more explicit, -1 if Set B is more implicit, 0 if guidance is stable across both sets.")


FORWARD_GUIDANCE_RANKING_PROMPT = """
 Your task is to analyze Federal Reserve statements on interest rate guidance across two periods and determine whether the guidance in Set B (later statement) is more explicit, more implicit (data-dependent), or remains at the same level of clarity compared to Set A (earlier statement).

This analysis will assess whether the FED is providing clearer signals on future rate decisions or shifting toward a more uncertain, data-dependent stance with fewer explicit commitments.

Evaluate the clarity and explicitness of interest rate guidance based on the following three aspects:

1.  Clarity of Forward Guidance on Interest Rates
    Compare how explicitly the FED communicates its expected rate path in Set A vs. Set B:

    - If Set B provides a clearer roadmap for future rate moves (e.g., specifying the likelihood of rate hikes, pauses, or cuts with a timeline or specific conditions), it suggests that guidance has become more explicit.
    - If Set B avoids clear forward guidance, shifts toward uncertainty, or emphasizes a data-dependent approach, it suggests that guidance has become more implicit.
    - If the level of specificity remains similar between Set A and Set B, guidance remains unchanged.

    Key Factors to Analyze:

        - Explicit references to future rate decisions: Does the FED specify what it plans to do (e.g., "We expect additional hikes may be needed")?
        - Degree of conditionality: Is the FED attaching clear conditions to its guidance, or is it becoming vaguer?
        - Use of uncertain vs. confident language: Are phrases like "likely to require" replaced by "will depend on incoming data"?

2. Change in Sentiment and Tone Regarding Policy Certainty
    Compare whether the FED's language reflects more certainty or more ambiguity in Set B vs. Set A:
        - If Set B expresses stronger conviction about the next policy move, guidance is becoming more explicit.
        - If Set B introduces uncertainty, stresses a "meeting-by-meeting" approach, or avoids committing to a policy path, guidance is becoming more implicit.
        - If the level of certainty remains unchanged, guidance is stable.

    Key Shifts to Look For:
        - Clear to vague language (e.g., "Further hikes are appropriate" → "Future decisions will be data-dependent" = More implicit)
        - Vague to clear language (e.g., "Policy adjustments will depend on conditions" → "We anticipate further rate increases" = More explicit)
        - Consistently explicit or vague language (No shift)

3. Change in Market Signaling and Policy Implications
    Determine whether the FED's messaging to financial markets has changed in terms of clarity:
        - If Set B provides more transparency and fewer uncertainties about rate moves, this suggests a shift toward more explicit guidance.
        - If Set B introduces ambiguity or places more emphasis on monitoring conditions before deciding, this suggests a shift toward more implicit guidance.
        - If the FED maintains a similar level of communication clarity, guidance is unchanged.

    Key Policy Signals to Compare:
        - Forward-looking language: Does Set B provide clearer expectations for market participants?
        - References to inflation or employment thresholds: Has the FED made rate decisions more conditional on uncertain economic developments?
        - References to external risks: Is the FED highlighting geopolitical or financial stability risks to justify uncertainty?

    Final Classification
        After evaluating these three dimensions, classify the change in interest rate guidance into one of the following categories:

        - More Explicit Guidance (The FED provides clearer signals on future rate decisions.)
        - More Implicit Guidance (The FED shifts toward uncertainty, data-dependent, or ambiguous guidance.)
        - Guidance Level Unchanged (The FED maintains a similar level of clarity in both statements.)
"""

class SummaryResponse(BaseModel):
    employment: str = Field(description="The summary of the employment topic")
    inflation: str = Field(description="The summary of the inflation topic")
    interest_rate_projections: str = Field(description="The summary of the interest rate projections topic")
    balance_sheet_projections: str = Field(description="The summary of the balance sheet projections topic")
    forward_guidance: str = Field(description="The summary of the forward guidance topic")

REASONING_SYNTHESIS_PROMPT = """
You are an expert macroeconomic analyst. You have been provided with a collection of detailed ranking reasonings. These reasonings were generated by comparing a specific 'reference document' (often referred to as 'Statement 2' or 'Set B' in the underlying individual ranking tasks) against several of its recent 'neighbor documents' (interchangeably 'Statement 1' or 'Set A').

Your task is to synthesize these individual comparisons and produce a concise, high-level summary. This summary must explain, for each major macroeconomic topic, how the **reference document, on average, ranked or was perceived compared to its recent neighbors.** This summary will ultimately serve as an explanation for an aggregated score assigned to the reference document.

The major macroeconomic topics to cover are:
1.  **Employment:** Synthesize findings from employment levels and employment dynamics.
2.  **Inflation:** Synthesize findings from inflation levels and inflation dynamics.
3.  **Interest Rate Projections:** Summarize the overall stance on interest rates.
4.  **Balance Sheet Projections:** Summarize the overall stance on the balance sheet.
5.  **Forward Guidance:** Summarize the clarity and nature of forward guidance.

For each of these five topics, your summary should:
*   Clearly state the general tendency. For example, was the reference document generally perceived as indicating a stronger/weaker labor market, accelerating/decelerating inflation, a more hawkish/dovish stance on interest rates, a more expansionary/contractionary balance sheet policy, or more explicit/implicit forward guidance when compared to the set of neighbor documents?
*   Briefly justify this overall assessment by highlighting the key themes, patterns, or a consensus view observed across the multiple provided reasonings and ranking scores for that topic. Avoid quoting extensively from the individual reasonings; instead, paraphrase, synthesize, and capture the essence of the comparisons.
*   The goal is to provide an overarching explanation of the reference document's positioning relative to its immediate historical context, based *only* on the provided reasonings.
*   You must provide summary in a way that is suitable for a investing professional.
*   You must refer to the reference document as 'This Press conference'
*   Do not mention things like 'Here is a high-level synthesis', write directly for the user. 
*   Your summary must not be more than 150 words. 

Input Data:
The ranking reasonings will be provided in a structured format (typically JSON). This structure contains, for each major topic and its sub-tasks (e.g., 'employment' -> 'level' and 'dynamics'), a list of comparisons. Each comparison includes:
    - A date of the neighbor document (the one that you should compare to) and the date of the reference document (the one that you should describe).
    - A numerical 'ranking' score (e.g., 1, -1, 0 based on the specific sub-task's criteria).
    - A textual 'reasoning' field explaining that specific ranking.
    - Reference document (the one that you should describe) is always refered to as 'Set B'.

Interpret the 'ranking' scores in the context of each sub-task's definition (e.g., for Employment Level, a ranking might mean 'Set B is stronger/weaker'; for Interest Rate Projection, 'Set B is more accommodative/restrictive'). Combine this with the textual 'reasoning' to form your summary for each major topic.

Example thought process for one topic (e.g., Employment):
If, across comparisons with 3 neighbor documents:
- For 'employment_level', the reference document was deemed to reflect a 'stronger' labor market in 2 comparisons (e.g., ranking = 1, reasoning supports strength) and 'similar' in 1 (e.g., ranking = 0).
- For 'employment_dynamics', it showed 'acceleration' in 1 comparison (e.g., ranking = 1), 'stabilization' in 1 (e.g., ranking = 0), and 'deceleration' in 1 (e.g., ranking = -1).
Your summary for the "Employment" topic might then state something like: 'Regarding employment, the reference document generally portrayed a somewhat stronger labor market than its recent predecessors, with mixed signals on momentum. While descriptions of overall labor market conditions often indicated improvement, the trajectory of employment dynamics showed less consistency, with some comparisons pointing to acceleration and others to stabilization or slight deceleration.'
"""