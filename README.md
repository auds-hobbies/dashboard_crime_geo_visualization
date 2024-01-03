# CRIME (SPATIAL) ANALYSIS 
Below is a list of my Data Science projects accomplished with tools such as python, HTML, CSS, Django and a number of algorithms etc:  

<br> 
<section>
<h3> Heart Risk Predictor --- Django Web App, Power BI Dashboard, and Excel Dashboard </h3> 
<p> A machine learning web app that helps doctors and nurses predict the heart risk of a patient. </p> 
<ul>
        <li> <b>Data Source(s): </b>San Francisco crime incidents; San Francisco shape files   </li>
        <li> <b>Data Science Techniques: </b>Classification   </li>
        <li> <b>Geo - Spatial Techniques: </b> spatial autocorrelation - WIP   </li>
        <li> <b>Approach (Methodology, Algorithms & Packages, Results): </b>Engineered new features with python; merged shape files with crime incidents data; ---------------------------- REMOVE ------------ data transforms using label encoder; feature selection with Random Forest feature importance etc. Compared multiple models using algorithms such as Logistic Regression, SVM, Random Forest etc, to arrive at Random Forest as the optimal model with accuracy ~ 82.6% and precision ~ 82.67% on the training set (<i>while ~80.2% and ~76.7% on the test set.</i>)      </li>
        <li> <b>Tools:</b> Python, Power BI (DAX)  </li>
        <li> <b>Outcomes / Outputs:</b> Power BI dashboard     </li>
</ul>
<p> The python code, pictures, video (demo), and/or a final report can be seen at the following links:
        <a href="https://github.com/auds-hobbies/p1_heart_risk_predictor" target="_blank"> GitHub(Python Code etc) </a>,
        <a href="https://www.youtube.com/watch?v=fBfwwSnnmyA"> YouTube (Demo)</a>,
        <a href="#report" target="_blank"> Report (PowerPoint)</a>
</p>
<div style="width: 350 px; float: left; height: 350 px;">
    <!-- Content for the blue div goes here -->
     <img src="https://github.com/auds-hobbies/dashboard_crime_geo_visualization/blob/main/github_crime_analysis_power_bi_dashboard_page1.png"  width = "250"  />
    <img src="https://github.com/auds-hobbies/dashboard_crime_geo_visualization/blob/main/github_crime_analysis_power_bi_dashboard_page2.png?raw=true"  width = "280"  />
     <img src="https://github.com/auds-hobbies/dashboard_crime_geo_visualization/blob/main/github_crime_analysis_power_bi_dashboard_page6.png?raw=true"  width = "280"  />
    
</div>
</section> 











# ---------- END OF TEST --------------- 



















# dashboard_crime_geo_visualization

<img src="dashboard_concept_v0.0.jpg" alt="DashboardConcept">

# CRIME ANALYSIS

<h3>Project Overview:</h3> 
The San Francisco police department takes note of crime incidents within their jurisdiction in terms of the date the crime occurred, the type of crime, and resolution et cetera. Other functions include having an oversight of crime per region in order to know how to allocate resources. This project seeks to provide insights into findings in the crime. These include: 

Insights and visualization on crime across the districts.
Identification of kinds of problems in the districts (statistical analysis). 
Identification of hot/cold spots.  

<br>
<section>
  <p> Analysis on vitals taken from admitted patients in order to determine the risks categories of admitted patients.</p>
    </p>
        <a href="https://github.com/auds-hobbies/dashboard_crime_geo_visualization" target="_blank"> GitHub </a>,
        <a href="https://www.youtube.com" target="_blank"> Twitter</a>,
        <a href="https://www.youtube.com" target="_blank"> YouTube</a>
    </p>   
  
    <div style="background-color: blue; width: 150 px; float: left; height: 150 px;">
    <!-- Content for the blue div goes here -->
     <img src="https://github.com/auds-hobbies/dashboard_crime_geo_visualization/blob/main/Screenshot%20crime%20analytics2.png?raw=true"  width = "300"  />
    <img src="https://github.com/auds-hobbies/dashboard_crime_geo_visualization/blob/main/Screenshot%20crime%20analytics2.png?raw=true"  width = "350"  />
    </div>
</section>

To be completed 
### Video 
Click on the picture below to view a video of the Excel Dashboard, Power BI Dashboard, and Heart Risk Predictor web app:

[![Watch the video](https://github.com/auds-hobbies/dashboard_crime_geo_visualization/blob/main/Screenshot%20crime%20analytics2.png)](https://youtu.be/pFVV-cahsBc) 


- Interactive Power BI Dashboard:
<img src="https://github.com/auds-hobbies/dashboard_crime_geo_visualization/blob/main/github_crime_analysis_power_bi_dashboard_page1.png " width="528"/> 



<h3>Data Collection and Processing:</h3>
Data: Data sources for this project include crime incidents reported within the San Francisco (SFO) area, location of police stations, and SFO shape files.  The crime data comprised 150 000 incidents. 

Entity Relationship: The crime data was merged with the shape files along districts. 

Data Preparation: Check for missing values and duplicates et cetera. Duplicated incident numbers with differing categories of crime were treated as one incident report. 

<h3>Analytical Techniques Applied:</h3>
Approach: This includes categorizing the kinds of crime incidents into 8 via text analysis, calculating summary statistics on the number and kinds of crime per district, and conducting spatial analysis (hot/cold spot) analysis across the districts. Algorithms used include but not limited to k-means clustering, DBSCAN et cetera. 

Crime Labels: A total of 8 labels were created on the category of crime reported, namely:

Theft: Comprising theft, larceny, vehicle theft, robbery, property theft etc
Assault – Vandalism: Comprising assault, vandalism etc
Missing_Suicide: missing person, kidnapping, suicide etc
Fraud: comprising fraud, embezzlement, forgery
Narcotics_Drunk_Sex: comprising sex offences,  prostitution, pornography, drinking under influence, drunkenness etc
Gambling_Warrants_Other: gambling etc
Noncriminal: recovered vehicle, non-criminal offences etc
Unknown: 
Statistical Analysis: The total number of crime incidents and kinds of crime were calculated per district.  


<h3>Results and Insights:</h3>

<img src="https://github.com/auds-hobbies/dashboard_crime_geo_visualization/blob/main/Screenshot%20crime%20analytics2.png " width="728"/> 


<h3>Future Developments:</h3>
Geo-Spatial Analysis: To be completed 

Hot/cold spot analysis
NOTE:  Work is in progress on this dataset and will be updated in due course.
