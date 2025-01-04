PROMPTS = [
    {
        "name": "generate_implicit_guidance",
        "prompt": """Your task is to analyze Analyze the Federal Reserve's minutes for the level of Forward Guidance, focusing on whether the guidance is Explicit or Implicit. Explicit guidance sets clear expectations for future policy moves, while implicit guidance is more vague and open-ended. 
            At first you must Examine the minutes, identifying the level of forward guidance (Explicit vs. Implicit) and assessing how specifically or vaguely future policy moves are communicated.
        Then you must propose specific changes to the language and structure of the minutes to make the forward guidance more implicit. Ensure these proposed adjustments align with the typical language, tone, and format used in Fed communications to maintain consistency with real-world practices.
        
        Output Structure:
        Start with your analysis of the minutes.
        After the analysis, include your strategies under the tag ##Strategies.
    
        FED Minutes to analyze:
        {minutes}
    
        Your response:
        """,
        "llm": 'gpt-4o'
    },
    {
        "name": "generate_explicit_guidance",
        "prompt": """
            You are a highly regarded expert in macroeconomics and central bank policy. 
            Your task is to analyze Analyze the Federal Reserve's minutes for the level of Forward Guidance, focusing on whether the guidance is Explicit or Implicit. Explicit guidance sets clear expectations for future policy moves, while implicit guidance is more vague and open-ended.
            At first you must Examine the minutes, identifying the level of forward guidance (Explicit vs. Implicit) and assessing how specifically or vaguely future policy moves are communicated.
            Then you must propose specific changes to the language and structure of the minutes to make the forward guidance more explicit. Ensure these proposed adjustments align with the typical language, tone, and format used in Fed communications to maintain consistency with real-world practices.
        
            Output Structure:
            Start with your analysis of the minutes.
            After the analysis, include your strategies under the tag ##Strategies.
        
            FED Minutes to analyze:
            {minutes}
        
            Your response: """,
        "llm": 'gpt-4o'
    },
    {
        "name": "generate_implicit_minutes",
        "prompt": """
            Generate a Federal Reserve minutes report with a focus on explicit forward guidance, clearly outlining expectations for future policy moves and their anticipated impacts.
            You will receive an analysis of the minutes and strategies how to make guidance more explicit. Follow them to generate the new minutes.
            
            ##Instructions
            - Report Structure & Length: The generated report should closely mirror the length, structure, and general style of the original minutes to maintain realism. Your report should contain all the dates, names and events from the original minutes.
            - Incorporate Explicit Guidance Strategies: Apply the previously generated strategies to enhance explicitness, clearly stating anticipated policy moves, economic projections, and their expected impacts on inflation, employment, and economic stability. Ensure the report communicates clear expectations and provides direct insights into future policy directions.
            
            YOU MUST generate a minutes of the same length and structure!
            
            Original Minutes:
            {minutes}
            
            Previously Generated Strategies for Implicit Guidance:
            {strategies}
            
            Generated Report:
        """,
        "llm": 'gpt-4o'
    },
    {
        "name": "generate_explicit_minutes",
        "prompt": """
            Generate a Federal Reserve minutes report with a focus on implicit forward guidance, subtly indicating policy direction without explicitly stating future moves. 
            You will receive an analysis of the minutes and strategies for making guidance more implicit. Follow them to generate the new minutes.
            
            ##Instructions
            - Report Structure & Length: The generated report should closely mirror the length, structure, and general style of the original minutes to maintain realism. Your report should contain all the dates, names and events from the original minutes.
            - Incorporate Implicit Guidance Strategies: Apply the previously generated strategies to the original minutes to enhance implicitness, using nuanced language to hint at economic outlook and possible policy directions without clearly stating them. Keep statements open-ended and focus on broad economic indicators, allowing for flexibility in interpretation.
            
            YOU MUST generate a minutes of the same length and structure!
            
            Original Minutes:
            {minutes}
            
            Previously Generated Strategies for Implicit Guidance:
            {strategies}
            
            Generated Report:
        """,
        "llm": 'gpt-4o'
    },
    {
        "name": "score_minutes",
        "prompt": """
            Your task is to analyze FED's minutes by the requested axes and provide scores.
            Please provide the analysis first and then output a json file where each axis is scored from 0 to 10.
            Also for each axis provide a short reasoning for your score.

            Your answer should be in format {{'axis': {{'score': 0, 'reasoning': 'reasoning'}}}}

            ##Axes:
            {axes}

            ##Minutes:
            {text}

            ##Your response:
        """,
        "llm": 'gpt-4o',
        "basic_axes": [
            "Forward Guidance (Explicit vs Implicit). Explicit guidance sets clear expectations for future policy moves, while implicit guidance is more vague and open-ended. 0 is for totally implicit guidance, 5 for mixed guidance, 10 is for totally explicit guidance.",
            "Certainty vs. Uncertainty: Certainty reflects the FOMC’s confidence level in their policy path, impacting market expectations. Clear signals of certainty or hesitancy provide valuable insight into policy stability. 0 is for totally uncertaince where fed doesn't know what will happen next, 5 is for some certainty, 10 is for totally certain.",
            "Economic Tone (Optimistic vs. Pessimistic): The FOMC's optimistic tone reflects its expectations regarding economic conditions, growth, and stability. 0 is pessimistic, when economy in a complete recession, 5 is for a balanced optimism, 10 is for an total optimism (usually recovery after hard crisis). ",
        ]
    },
    {
        "name": "score_statements",
        "prompt": """
            Your task is to analyze FED's statements by the requested axes and provide scores.
            Please provide the analysis first and then output a json file where each axis is scored from 0 to 10.
            Also for each axis provide a short reasoning for your score.

            Your answer should be in format {{'axis': {{'score': 0, 'reasoning': 'reasoning'}}}}

            ##Axes:
            {axes}

            ##Statement:
            {text}

            ##Your response:
        """,
        "llm": 'gpt-4o',
        "basic_axes": [
            "Guidance (Explicit vs. Implicit): Measures the clarity of future policy signals. Explicit guidance provides clear indications of future actions, while implicit guidance leaves more to interpretation.",
            "Uncertainty (Certain vs. Uncertain): Reflects the confidence in the economic forecast and policy actions. Certain language suggests a stable outlook, while uncertainty hints at potential adjustments due to economic unpredictability.",
            "Direction (Rate Hikes vs. Cuts): Indicates the Fed’s intent regarding interest rates. Rate hikes suggest tightening, while rate cuts imply easing.",
            "Economic Outlook (Positive vs. Negative Tone): Captures sentiment about economic growth, inflation, and employment. Positive language indicates optimism, while negative language may signal caution.",
            "Commitment to Policy (Firm vs. Flexible Commitment): Assesses the strength of commitment to future actions. Firm language suggests a rigid stance, while flexible language implies potential responsiveness to new data.",
            "Inflation Targeting (Tolerant vs. Intolerant Language): Evaluates the Fed’s tolerance for above-target inflation. Tolerant language suggests prioritizing employment or growth, while intolerance indicates aggressive action against inflation.",
            "Global Economic Conditions (Isolationist vs. Internationally Minded): Examines references to foreign influences. Isolationist language indicates a domestic focus, while internationally minded language suggests awareness of global economic factors.",
            "Market Reassurance (Calm vs. Cautionary Language): Assesses the tone intended for market stability. Calm language aims to reassure markets, while cautionary language may lead to market caution or volatility."
        ]
    },
    {
        "name": "generate_statements",
        "prompt": """
            You are an expert in macroeconomics and Federal Reserve (Fed) policy analysis.
            Your task is to revise a given Fed statement to align it more strongly with the specified axis and requested change. 
            For example, if asked to make the statement more Explicit on the Guidance axis, you should adjust the statement to provide clearer policy signals.
            Output 

            ##Axes
            {axes}

            ##Statement:
            {text}

            ##Requested Change:
            {change}

            """,
        "basic_axes": [
            "Guidance (Explicit vs. Implicit): Measures the clarity of future policy signals. Explicit guidance provides clear indications of future actions, while implicit guidance leaves more to interpretation.",
            "Uncertainty (Certain vs. Uncertain): Reflects the confidence in the economic forecast and policy actions. Certain language suggests a stable outlook, while uncertainty hints at potential adjustments due to economic unpredictability.",
            "Direction (Rate Hikes vs. Cuts): Indicates the Fed’s intent regarding interest rates. Rate hikes suggest tightening, while rate cuts imply easing.",
            "Economic Outlook (Positive vs. Negative Tone): Captures sentiment about economic growth, inflation, and employment. Positive language indicates optimism, while negative language may signal caution.",
            "Commitment to Policy (Firm vs. Flexible Commitment): Assesses the strength of commitment to future actions. Firm language suggests a rigid stance, while flexible language implies potential responsiveness to new data.",
            "Inflation Targeting (Tolerant vs. Intolerant Language): Evaluates the Fed’s tolerance for above-target inflation. Tolerant language suggests prioritizing employment or growth, while intolerance indicates aggressive action against inflation.",
        ],
        "basic_changes": [
            "Make the statement more Explicit in Guidance, providing clear indications of future policy.",
            "Make the statement more Implicit in Guidance, leaving future policy open to interpretation.",
            "Make the statement more Certain, reflecting confidence in economic forecasts and actions.",
            "Make the statement more Uncertain, highlighting risks and the potential for unexpected changes.",
            "Make the statement more Firm in Commitment, emphasizing a steadfast approach to current policy.",
            "Make the statement more Flexible in Commitment, indicating decisions will adapt to future data.",
            "Make the statement more Tolerant of Inflation, prioritizing growth and employment over strict inflation control.",
            "Make the statement more Intolerant of Inflation, prioritizing inflation containment over other goals.",
            "Make the statement more Positive in Economic Outlook, expressing optimism about economic conditions.",
            "Make the statement more Negative in Economic Outlook, reflecting caution or pessimism about the economy."
        ]
    },
    {
        "name": "generate_press_conferences",
        "prompt": """
            You are an expert in macroeconomics and Federal Reserve (Fed) policy analysis.
            Your task is to revise a given Fed statement to align it more strongly with the specified axis and requested change. 
            For example, if asked to make the statement more Explicit on the Guidance axis, you should adjust the statement to provide clearer policy signals.
            Output 

            """
    }
]