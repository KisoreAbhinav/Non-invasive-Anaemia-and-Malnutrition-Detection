## GOAL:

### To build an ML based Health Tech sys for screening and risk assesment of anaemia and malnutrition in children and pregnant women NONINVASIVELY AND THE DEVICE MUST BE MOBILE

> So we have a raspberry pi 16gb and camera and mic and speakers and shit

> > Going to have a multimodal architecture, no single model takes in raw data and gives a decision, we stagger it and obtain only classes from individual services and the final classification is done by us using custom weights or another ML model so essentially feature inputs to different models gives us another feature vector that goes into another model

> > > Need a model for eyes, nail, tongue ka pallor, also need for that foot edema which is fine for prego women but bachho mai nahi then weight-age-height waale features ka processing and then all that being fed into another model

> > > > also need some good datasets to train them or use pretrained shit otherwise nahi hora kaam bc

### Children

- Stage 1 -> MUAC screening

No models required for this one
Can override if critical
Skip other tests

- Stage 2 -> Edema check
  3sec thumb pressure
  pits/doesnt pit
  if edema -> automatically diagnose as severely acute malnourished
  ovverrides MUAC and weight-for-height
  critical ovverride by WHO protocols

(my proposed solution) -- we ask the user to apply pressure and display a countdown on screen and then we take a snap and run it through a NN, and its foolproof because agar pit is not visible then it mustve bounced back in the meantime and if the pit is still visible then its guaranteed edema matlab itni der ke baad bhj agar nhi gaya hai to matlab its fucked up (as we saw on google how the pits are actually fucking prominent and like maa chudi padi hai logo ki but theek hai). this eliminates the use of time-series data or the "difference in data" going to the model.

- Stage 3 -> Weight, height and shit
  agar weight machine hai to fir heartrate monitor bhi hona hi chahiye bhenchod ye kya baat hui matlab kuch bhi

Weight-height
Height-age
weight-age

trifecta samajh ke chalo

represent as Z-scores (gaussian dist) -- tells us if its acute or chronic and so we obtain treatment method

- Stage 4 -> Clinical masti
  hair changes, skin flaking, loss of appetite
  CNN for all that shit, seedha yes-no sawaal ka jawaab de bc koi camera ki zarurat nahi
  "do you notice flaking on your skin?; yeah? awesome"

- Stage 5 -> WHO Appetite Test -- give food, observe

- Stage 6 -> Treatment and shit lmao, baadme mai dekhenge

### Women

- Stage 1 - MUAC, just single threshold, no color coding

- Stage 2 - Pre-prego BMI, set once and then use throughout

- Stage 3 - gestational weight gain tracking, time series data of weight vs each visit to the gyno

- Stage 4 - weird bc, fym tip of pubic bone to uterus. Fundal height measurement in cm ~ roughly equal to weeks of gestation; just used for screening

- Stage 5 - Same as clinical data for kids EXCEPT edema is not a clear signal here so

- Stage 6 - Proactive step, needed
  - Haemoglobin
  - Serum Ferritin
  - Albumin
  - Blood glucose
  - Folate, B12
  - Calcium Vitamin D Zin
  - Reactive in kids, proactive in prego
