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
