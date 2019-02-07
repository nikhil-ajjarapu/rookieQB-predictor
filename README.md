# rookieQB-predictor

Technical Description: 

Uses machine learning to estimate rookie QB performance. 

(Currently) uses 4 factors:
* QBR in college, as determined by ESPN.
* Strength of the Offensive Line in college, as determined by numerous statistics calculated by [Football Outsiders](https://www.footballoutsiders.com/info/methods) (detailed by the link attached).
* Strength of the Defense overall in college, as determined by Football Outsiders as well. 
* Strength of the Offensive Line of the team the QB is drafted to using data from the previous year by Football Outsiders

These four factors are used to make a prediction of the Rookie QBs' ANY/A (Adjusted Net Yards/Attempt), a statistic that is [highly correlated](https://www.sportingcharts.com/dictionary/nfl/adjusted-net-yards-per-pass-attempt.aspx) with winning and QB skill and success in general. 

Abstract:
In 1980, Bill James, a night watchman at a local factory began to publish articles and books about different statistical analyses which could be used to describe in-game activity. His methods, dubbed “sabermetrics”, were proven to be significantly more reliable than other methods like the human eye test and gut feelings, which were used back then to analyze the game. Currently, a common problem facing American football teams in the National Football League (NFL) is finding the perfect quarterback (QB) for their team. The quarterback is arguably the most important position on any given team, and often the success of the QB is one of, if not, the most correlated with the success of the team. At such a important position, it is shocking to see that QBs even today are judged primarily based on the eye test, which often leads to many “busts”, or unsuccessful QBs, being drafted, which can set the NFL team back years. In this project, I have attempted to predict how well rookie QBs will fare in their first season based on two general factors: their performance in a college setting, and the strength of the NFL team they are drafted to. Functioning at a 73% accuracy rate, my machine learning model can help provide NFL teams valuable advice about whether to draft a certain QB or not, and ultimately if they would be a good fit for their team. 
