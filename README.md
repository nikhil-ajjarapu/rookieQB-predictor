# rookieQB-predictor
Uses machine learning to estimate rookie QB performance. 

(Currently) uses 4 factors:
* QBR in college, as determined by ESPN.
* Strength of the Offensive Line in college, as determined by numerous statistics calculated by [Football Outsiders](https://www.footballoutsiders.com/info/methods) (detailed by the link attached).
* Strength of the Defense overall in college, as determined by Football Outsiders as well. 
* Strength of the Offensive Line of the team the QB is drafted to using data from the previous year by Football Outsiders

These four factors are used to make a prediction of the Rookie QBs' ANY/A (Adjusted Net Yards/Attempt), a statistic that is [highly correlated](https://www.sportingcharts.com/dictionary/nfl/adjusted-net-yards-per-pass-attempt.aspx) with winning and QB skill and success in general. 
