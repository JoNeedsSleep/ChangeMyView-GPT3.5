# ChangeMyView-GPT3.5
 Based on Chenhao Tan's 2016 paper [Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions](https://chenhaot.com/papers/changemyview.html), this project leverages the semantic analysis capabilities of GPT-3.5 to identify tonal attributes that influence the persuasiveness of textual content.

 After reading the original paper, I was very fascinated by the overarching goal: finding subtle linguistic trends that hold predictive power for persuasiveness. Given that the features extracted in the original paper were of a lexical nature, I was interested to use GPT-3.5 to perform some semantic classification and see how they relate to both GPT-3.5’s predicted persuasiveness and their actual delta vote. Due to a lack of processing power, I used around 500 pairs of data from the training set, totaling to 1086 data points.

 With some [prompt engineering](https://github.com/JoNeedsSleep/ChangeMyView-GPT3.5/blob/main/Prompt_Template.txt) I asked GPT-3.5 to label the formality, subjectivity, optimistic vs. cynical tone, extremity, and lexical density from -1.0 to 1.0. The choice of these tonal labels and the wording of these criteria is from my personal knowledge of textual features. Based on these tonal observations, I asked it to make a judgement from 0.0 to 1.0 on the overall persuasiveness.


Plotting a violin graph gave a more intuitive visualization of the data distribution. Here the darker contour represents the positive data and the lighter contour the negative. In accordance with reddit most posts were informal, subjective with a heavy dose of cynicism. More positive posts were optimistic and high in lexical density. Notably, the posts getting the delta votes seemed to be more polarized on the extremity scale.
 
![attribute_violin](https://github.com/JoNeedsSleep/ChangeMyView-GPT3.5/assets/39445027/dfccbc31-0f2e-453f-a46e-9dbcaf11d5c8)

While I was confident that GPT-3.5 could do a pretty good job extracting semantic data, I was curious to see how good the persuasiveness predictions are since they involved higher level reasoning. Hence, I employed a logistic regression model to predict the likelihood of an argument being classified as 'positive,' which, in this context, corresponds to receiving a delta vote on a forum. This roughly approximates to the persuasiveness score. 
Below is a 2D kernel density estimation (KDE) graph overlaid with a regression trend line. There is a weakly positive correlation between the persuasiveness scores generated by GPT 3.5 and the predicted probability generated by the logistic regression model, suggesting that the concept of persuasiveness demands a richer interpretation and more comprehensive extraction of textual data. 
![logreg_KDE_scatter](https://github.com/JoNeedsSleep/ChangeMyView-GPT3.5/assets/39445027/7a3723d3-628d-49a8-a36f-fba82d996caa)
