# AI-Dataviz-Exploration

Sharing an output from my experimentation with agentic AI for data insights democratization.

It’s a simple agentic app for plain-language exploration of a dataset from CultureX, containing corporate culture values across companies from different industries, measured through anonymous Glassdoor reviews between Jan 1, 2023, and Apr 4, 2025.

It consists of several agents (powered by the GPT-5-mini model for inference), each responsible for a different aspect of the insight-generation process and using inputs from other agents:

* one agent plans the analysis steps based on the user’s request, data characteristics, and broader context,
* another turns this plan into runnable code,
* another fixes potential errors in the code based on error messages,
* another creates code for accompanying analyses to support explanation of the dataviz,
* and another transforms everything into a concise, clear narrative for a non-technical audience.

It runs on a public dataset from *CultureX* with corporate culture values across companies in different industries, measured through anonymous Glassdoor reviews between Jan 1, 2023, and Apr 4, 2025.

It usually provides good outputs even when you use really simple plain language - but based on my experiments, it helps if you know what kind of dataviz and insights you want and spell that out for the agents 😉

Happy exploring 🕵️‍♀️

P.S. Since the app is hosted on Streamlit Community Cloud, it doesn’t stay awake continuously. If it hasn’t been used recently, you may need to wake it up and wait a few minutes.

P.P.S. As usual, genAI can make mistakes, so don’t trust the outputs blindly - always double-check 😉

----
**Update**: After Google's release of Gemini 3 Flash Preview, which demonstrates a superior performance-to-price ratio, I replaced GPT-5-mini with this model. Its higher "intelligence" is readily apparent in the quality of the app's outputs - specifically through better chart selection and more nuanced data interpretations. It's a perfect example of the ongoing commoditization of "intelligence."
