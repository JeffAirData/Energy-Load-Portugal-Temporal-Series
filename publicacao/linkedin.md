Imagem principal: `docs/img/painel_resumo.png` não existe neste projeto. Use o carrossel `docs/img/cartao_1.png` a `cartao_5.png` (1080x1080), nessa ordem, ou `docs/img/03_modelos.png` como imagem única.

---

I've been going back through my old GitHub projects, and there's always one that's the apple of your eye. Mine is the time series of Portugal's electricity demand.

The question is simple: can you guess how much electricity a country will use tomorrow?

I had 18 months of hourly data (13,104 readings) and tried to answer it step by step:

- I looked at the shape of the data first. Demand follows a daily rhythm and a weekly one, and the quietest days use about 13% less than the rest.
- I set the last 5 months aside and never showed them to the models. That's how you know a model really learned something.
- The lazy guess, "tomorrow will be like today", is off by about 356 MW.
- A model that knows the weekly rhythm (SARIMA) or learns from the last 7 days (KNN, Random Forest) is off by about 105 MW. That's around 2% of the average load, one third of the lazy error.
- A score called AIC (it rewards a model for fitting the data and punishes it for being complicated) dropped from 7,773 to 7,191 when I added the weekly cycle. The missing piece was the week.
- Plot twist: 7 days ahead, these simple models barely beat "same weekday last week". Knowing when a model stops helping is part of the job.

One honest catch: the dates in this dataset look shifted from the real calendar (the quietest days show up as Tuesday and Wednesday, and there is no dip on public holidays). The weekly pattern is intact, so the comparison holds, but I treated weekdays as labels and left holidays out.

Data science can be fun, so I turned it into a small game: you see the last 7 days and guess tomorrow's demand. Can you beat the models?

👉 https://jeffairdata.github.io/Energy-Load-Portugal-Temporal-Series/

Code, notebook and results: https://github.com/JeffAirData/Energy-Load-Portugal-Temporal-Series

#DataScience #TimeSeries #Forecasting #Python #DataAnalytics #OpenData
