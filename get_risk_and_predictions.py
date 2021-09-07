'''
Version: optimizing on daily difference with death correction and SG_Filter, optimizing with vacc data as well. Added an exp model -> Used when the growth rate is negative.
This code has been tested on Python3.6 and can an hour to execute.
In this script, we calculate the intensity score and predictions (cases and deaths) for COVID-19 in different regions.
The base tables used are rto.daily_global, rto.daily_us, rto.daily_india, rto.country_population, rto.population_us and rto.population_india_states.

The following tables are updated using this script. 
regional_intensity_profiling
LG_predicted_cases
LG_predicted_deaths
regional_intensity_profiling_future
regional_intensity_predictions
intensity_fixed_params
vacc_current
vacc_predictions

Before running the script, note down the latest date in rto.regional_intensity_profiling using the below mentioned command. 
This tells us the date till which we have calculated the intensity scores.

select cast(max("date") as date) from rto.regional_intensity_profiling

After successful exection of the script, "Script executed Successfully!" should be printed out. In case of any errors, revert back using the below script (update the date):

delete from rto.regional_intensity_profiling where "date">'2021-08-08';
delete from rto.LG_predicted_cases where date_of_calc >'2021-08-08';
delete from rto.LG_predicted_cases_with_conf where date_of_calc >'2021-08-08';
delete from rto.LG_predicted_deaths where date_of_calc >'2021-08-08';
delete from rto.regional_intensity_profiling_future where date_of_calc >'2021-08-08';
delete from rto.regional_intensity_predictions where date_of_calc >'2021-08-08';
delete from rto.intensity_fixed_params where date_of_calc > '2021-08-08';
delete from rto.LG_predicted_cases_params where date_of_calc > '2021-08-08';
delete from rto.LG_predicted_deaths_params where date_of_calc > '2021-08-08';
'''
import warnings 
#warnings.filterwarnings("ignore", category=RuntimeWarning).
import logging
import datetime
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as optimize
import teradatasql
import getpass
import datetime
import sys
import os
from teradataml import create_context,remove_context,copy_to_sql,DataFrame
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error
import copy
import seaborn as sns

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
fh = logging.FileHandler('rto_script.log')
fh.setFormatter(formatter)
logger.addHandler(fh)
#logger.info("starting")
#exit()

'''def delete_entries(date_of_calc):

    """delete from rto.regional_intensity_profiling where "date">'{}';
    delete from rto.LG_predicted_cases where date_of_calc >'{}';
    delete from rto.LG_predicted_cases_with_conf where date_of_calc >'{}';
    delete from rto.LG_predicted_deaths where date_of_calc >'{}';
    delete from rto.regional_intensity_profiling_future where date_of_calc >'{}';
    delete from rto.regional_intensity_predictions where date_of_calc >'{}';
    delete from rto.intensity_fixed_params where date_of_calc > '{}';
    delete from rto.LG_predicted_cases_params where date_of_calc > '{}';"""

    connection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
    cur = connection.cursor()
    cur.execute(sql_command)
    res = cur.fetchall()
    cur.close()'''



def func(t, A, lamda): # y = A*exp(lambda*t)
    y = A*np.exp(lamda*t)
    return y
ROLLING_MEAN_FOR_GROWTH_CALC = 0
def cost_function(params):
    #WINDOW_RISK = 14
    y = func(np.arange(0,WINDOW_RISK,1),params[0],params[1])
    #print("y",y,D["JHU_ConfirmedCases.data"].diff().values[-WINDOW_RISK:])
    assert ROLLING_MEAN_FOR_GROWTH_CALC ==0 or ROLLING_MEAN_FOR_GROWTH_CALC ==1

     
    Ddiff = D.diff().fillna(value=D.diff().mean()) # in case the first value is NAN
    if ROLLING_MEAN_FOR_GROWTH_CALC ==0:
        return np.sum((y - Ddiff.values[-WINDOW_RISK:])**2)
    elif ROLLING_MEAN_FOR_GROWTH_CALC ==1:
        return np.sum((y - Ddiff.rolling(window=14).mean().values[-WINDOW_RISK:])**2)
def min_max_scaler_p(d):
    d = np.array(d)
    _max = np.percentile(d,95)
    _min = np.percentile(d,5)
    r = (d-_min)/(_max-_min)
    r[r>1] = 1
    r[r<0] = 0
    return r
def min_max_scaler(d):
    return (d-min(d))/(max(d)-min(d))
def get_risk(data,weights):
    risk = 0
    for d,w in zip(data,weights):
        risk = risk + d*w
    risk = risk / sum(weights)
    return risk

TD_countries = ["Argentina","Australia","Austria","Belgium","Brazil","Canada","Chile",
                "China","Colombia","Czech Republic","Denmark","Ecuador","Egypt","Finland",
                "France","Germany","Spain","India","Indonesia","Ireland","Israel","Italy",
                "Japan","Malaysia","Mexico","Netherlands","New Zealand","Norway","Pakistan",
                "Peru","Philippines","Poland","Russia","Saudi Arabia","Singapore",
                "South Korea","Sweden","Switzerland","United Arab Emirates","United Kingdom","United States"]
TD_regions = ["Argentina","Australia","Austria","Belgium","Brazil","Canada","Chile",
                "China","Colombia","Czech Republic","Denmark","Ecuador","Egypt","Finland",
                "France","Germany","Spain","India","Indonesia","Ireland","Israel","Italy",
                "Japan","Malaysia","Mexico","Netherlands","New Zealand","Norway","Pakistan",
                "Peru","Philippines","Poland","Russia","Saudi Arabia","Singapore",
                "South Korea","Sweden","Switzerland","United Arab Emirates","United Kingdom","United States","Maryland","Illinois","New York","Georgia","California"
"Santa Clara",
"San Diego",
"Los Angeles",
"Cook",
"Lexington",
"Travis",
"Dallas",
"King",
"Wake",
"Fulton"]

TD_regions =    ["Argentina","Australian Capital Territory",
                  "Austria","Bengaluru Urban","Brazil","China","Cook","Czech Republic",
                  "Dallas","Denmark","Egypt","Finland","France","Fukuoka","Fulton","Germany",
                  "Gurugram","Indonesia","Ireland","Italy","King","Lexington","Los Angeles","Malaysia",
                  "Mexico","Mumbai","Netherlands","New South Wales","New York","Osaka","Pakistan","Poland",
                  "Pune","Russia","San Diego","Santa Clara","Saudi Arabia","Singapore","South Korea","Spain",
                  "Sweden","Switzerland","Taiwan","Telangana","Tokyo","Travis","United Arab Emirates",
                  "United Kingdom","Wake"]



def get_sql_df(sql_command):
    connection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
    cur = connection.cursor()
    cur.execute(sql_command)
    res = cur.fetchall()
    cur.close()
    df = pd.DataFrame(res,columns=np.array(cur.description)[:,0])
    return df



################################################# Fetch data from vantage #########################################################

# fetch data from vantage
print("Fetching data from Vantage")

import teradatasql
connection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
cur = connection.cursor()

cur.execute("""
select main_data.*,vacc_data.pred_vacc_perc as vacc_perc,mob.avg_mob from 
(
    select "date","Country/Region",country_of_state,confirmed,deaths,pop_country as population from
    (select "date",
    case
    when "Country/Region" = 'Korea, South' then 'South Korea' 
    when "Country/Region" = 'Czechia' then 'Czech Republic'
    when "Country/Region" = 'US' then 'United States'
    when "Country/Region" = 'Taiwan*' then 'Taiwan' 
    else "Country/Region" end as "Country/Region",
    cast('' as varchar(30)) as country_of_state,
    sum(confirmed) as confirmed,sum(deaths) as deaths from rto.daily_global group by 1,2,3) a
    left join rto.country_population
    on country = "Country/Region"

    union all

    select "date",Province_state,country_of_state, confirmed, deaths, population from
    (select "date",Province_state,country_region as country_of_state,sum(confirmed) as confirmed,sum(deaths) as deaths from rto.daily_us group by 1,2,3) a
    left join (select state,sum(population) as population from rto.population_us group by 1) b
    on Province_state = state

    union all

    select "date",a.state,country_of_state, confirmed, deaths, population from
    (select "date",state,'India' as country_of_state,sum(confirmed) as confirmed,sum(deaths) as deaths from rto.daily_india_districts group by 1,2,3) a
    inner join (select state,population from rto.population_india_states where state = 'Telangana') b
    on a.state = b.state

    union all

    select "date",a.district,country_of_state, confirmed, deaths, population from
    (select "date",district,'India_district' as country_of_state,sum(confirmed) as confirmed,sum(deaths) as deaths from rto.daily_india_districts group by 1,2,3) a
    inner join (select district,population from rto.population_india_districts where district <>  'Hyderabad') b
    on a.district = b.district

    union all 

    select "date",county as "Country/Region",'US_county' as country_of_state,confirmed,deaths,population as population from rto.daily_us
    inner join
    (
        select * from 
        (select distinct(county_district) as county1,state_province,city,country_region from rto.td_sites where country_region='US') a
        inner join 
        (select * from rto.population_us) b
        on a.county1 = b.county and state_province = state
    ) a
    on admin2=county and province_state=state

    union all

    select "date",province_state, country_region as country_of_state,confirmed, deaths, popTotal as population
    from (select * from rto.stg_daily_global_province 
    where country_region in ('Australia','Japan') and province_state in (select state_province from rto.td_sites) ) m
    inner join rto.population_global
    on province_state = location

) main_data

left join (
select distinct "Country/Region",country_of_state, pred_vacc_perc from rto.regional_intensity_pred_with_vacc_view 
where date_of_calc = (select max(date_of_calc) from rto.regional_intensity_pred_with_vacc_view) and date_of_calc = "date"
) vacc_data
on main_data."Country/Region" = vacc_data."Country/Region" and main_data.country_of_state = vacc_data.country_of_state

left join rto.mobility_avg_view mob
on main_data."Country/Region" = mob."Country/Region"

order by 1;
""")
res = cur.fetchall()
cur.close()


df_cases = pd.DataFrame(res,columns=np.array(cur.description)[:,0])
df_cases["date"] = pd.to_datetime(df_cases["date"])
df_cases = df_cases.fillna(value=df_cases["population"].mean())
df_cases = df_cases.fillna(value=df_cases["population"].mean())
pop = df_cases["population"].values
pop[pop==0]= df_cases["population"].mean()
df_cases["population"] = pop
df_cases
    

    
##################################################################################################################################
logger.info("Data Fetched")

################################################## Calculate Risk ################################################################

# Calculate Risk
ROLLING_MEAN_FOR_GROWTH_CALC = 1
print("Calculating Risk")
conection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
cur = connection.cursor()
cur.execute("""select cast(max("date") as date)+1 from rto.regional_intensity_profiling where "Country/Region"='Zimbabwe'""")
max_date_region_int_tab = cur.fetchall()[0][0]
if type(max_date_region_int_tab) == datetime.date:
    max_date_region_int_tab = max_date_region_int_tab.strftime("%Y-%m-%d")
elif max_date_region_int_tab == None:
    max_date_region_int_tab = '2020-10-17'
if len(sys.argv) == 2:
    max_date_cases_tab = datetime.datetime.strptime(sys.argv[1],"%Y-%m-%d").date()
else:
    cur.execute("""
    select min(max_date) from
    (
    select max("date") as max_date from rto.daily_global 
    union all
    select max("date") as max_date from rto.daily_us
    --union all 
    --select max("date") as max_date from rto.daily_india
    union all 
    select max("date") as max_date from rto.daily_india_districts
    union all
    select max("date") as max_date from rto.stg_daily_global_province
    )as tmp;""")
    max_date_cases_tab = cur.fetchall()[0][0]
if type(max_date_cases_tab) == datetime.date:
    max_date_cases_tab = max_date_cases_tab.strftime("%Y-%m-%d")

cur.close()

#max_date_region_int_tab = '2020-10-18' # CHANGE THIS OR COMMENT IT IF NEEDED TO RECALCULATE
#max_date_cases_tab = '2021-02-27' # CHANGE THIS OR COMMENT IT IF NEEDED TO RECALCULATE

WINDOW_RISK = 14

print(max_date_region_int_tab,max_date_cases_tab)
if max_date_cases_tab>=max_date_region_int_tab:
    dates_to_calc= pd.date_range(start=max_date_region_int_tab, end = max_date_cases_tab)
    print(dates_to_calc)
    
    REGIONS = pd.DataFrame([],columns=['Country/Region', 'country_of_state', 'population', 'date', 'is_TD',
           'growth_rate', 'growth_rate_deaths', 'Re', 'total_cases',
           'total_cases_per_M', 'daily_cases', 'daily_cases_per_M', 'daily_deaths',
           'daily_deaths_per_M'],dtype=object)


    # to remove any partially calculated values
    conection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
    sql_command = """delete from rto.regional_intensity_profiling where "date"='{}';""".format(dates_to_calc[0].date().strftime("%Y-%m-%d"))
    cur.close()

    for date_to_calc in dates_to_calc:
        total_cases = []
        daily_cases = []
        daily_deaths = []
        growth_rates = []
        growth_rates_deaths = []
        
        A = 1000
        lamda = 0.001

        regions = df_cases[["Country/Region","country_of_state","population"]].drop_duplicates(subset=["Country/Region","country_of_state"]).sort_values(by="Country/Region")
        #regions = regions[regions["Country/Region"]=='New York'].query("country_of_state == 'US_county'")
        
        max_date_cases = df_cases["date"].values[-1]
        regions["date"] = date_to_calc.date().strftime("%Y-%m-%d")#max_date_cases
        IS_TD = []
        print(date_to_calc)
        for row in regions[["Country/Region","country_of_state","population"]].values:

            region,country_of_state,pop = row
            #print(region,country_of_state,pop)
            if region in TD_regions:
                IS_TD.append(1)
            else:
                IS_TD.append(0)

            #region_data = df_cases[df_cases["Country/Region"]== region].query("country_of_state == '"+str(country_of_state)+"'")
            region_data = df_cases[df_cases["Country/Region"]== region].query("country_of_state == '"+str(country_of_state)+"' and date <= '"+date_to_calc.date().strftime("%Y-%m-%d")+"'")

            D = region_data["confirmed"]
            initial_guess = [A,lamda]
            result = optimize.fmin(cost_function,initial_guess,maxfun=1000,maxiter=1000)
            A_,growth_rate = result 
            growth_rates.append(growth_rate)
            print(region,country_of_state,growth_rate)


            D = region_data["deaths"]#D = COUNTRY_DATA[country]["JHU_ConfirmedDeaths.data"]
            initial_guess = [A,lamda]
            result = optimize.fmin(cost_function,initial_guess,maxfun=1000,maxiter=1000)
            A_,growth_rate = result 
            growth_rates_deaths.append(growth_rate)

            total_cases.append(float(region_data["confirmed"].values[-1]))
            daily_cases.append(region_data["confirmed"].diff().values[-WINDOW_RISK:].mean())
            daily_deaths.append(region_data["deaths"].diff().values[-WINDOW_RISK:].mean())
        growth_rates = np.array(growth_rates)
        growth_rates_deaths = np.array(growth_rates_deaths)
        growth_rates[growth_rates<-0.5] = -0.5 # clip the negative value
        growth_rates_deaths[growth_rates_deaths<-0.5] = -0.5 # clip the negative value
        regions["is_TD"] = IS_TD
        regions["growth_rate"] = growth_rates
        regions["growth_rate_deaths"] = growth_rates_deaths
        Re = 1 + growth_rates*5
        Re[Re<0] = 0
        regions["Re"] = Re
        regions["total_cases"] = total_cases
        regions["total_cases_per_M"] = 1000000*regions["total_cases"]/regions["population"]
        daily_cases = np.array(daily_cases)
        daily_cases[daily_cases<0]=0
        daily_deaths = np.array(daily_deaths)
        daily_deaths[daily_deaths<0]=0
        regions["daily_cases"] = daily_cases
        regions["daily_cases_per_M"] = 1000000*regions["daily_cases"]/regions["population"]
        regions["daily_deaths"] = daily_deaths
        regions["daily_deaths_per_M"] = 1000000*regions["daily_deaths"]/regions["population"]
        REGIONS = pd.concat((REGIONS,regions),axis=0)
    #REGIONS = REGIONS.dropna()
    create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
    copy_to_sql(df=REGIONS,table_name="regional_intensity_profiling",schema_name="rto",if_exists="append",primary_index="Country/Region")
    remove_context()
else:
    #print("Script already executed till the latest date")
    #sys.exit()
    print("rto.regional_intensity_profiling already updated")
#REGIONS_CURRENT = REGIONS

##################################################################################################################################
logger.info("rto.regional_intensity_profiling' done. Risk Calculated till the latest date.")

###################################################### for prediction modeling #####################################################

# for prediction modeling

from scipy.optimize import Bounds
from scipy import stats

forget_factor = 0.9
WINDOW = 30
def get_predictions_sigmoid(x,alpha,lamda = 1,beta = 0, gamma = 1.):
    cases = (lamda / (1 + gamma*np.exp(-alpha*(x-beta)))**(1./gamma))
    return cases
def get_predictions_sigmoid_exp(t, A, lamda): # y = A*exp(lambda*t)
    y = A*np.exp(lamda*t)
    return y
def cost_predictions(params):
    global actual,WINDOW
    y = get_predictions_sigmoid(np.arange(0,len(actual),1),params[0],params[1],params[2],params[3])
    
    # in case fitting on diff is required
    y = np.diff(y)
    actual2= np.diff(actual)
    actual2[actual2<0]=0
    #actual = np.diff(actual)
    f = [forget_factor**i for i in range(len(actual2))][::-1]
    # [... 0.99^3 0.99^2 0.99]
    
    return np.sum(f[-WINDOW:]*(y[-WINDOW:] - actual2[-WINDOW:])**2)

def cost_actual(params,model_type = 'LG'):
    global actual,window_for_averaging
    actual2= np.diff(actual)
    actual2[actual2<0]=0
    assert model_type == 'LG' or model_type == 'exp'
    win_actual = window_for_averaging#2*window_for_averaging
    den = float(np.average(actual2[-win_actual:]))
    

    
    if model_type == 'LG':
        win_actual = window_for_averaging#2*window_for_averaging
        y = get_predictions_sigmoid(np.arange(0,len(actual)+365,1),params[0],params[1],params[2],params[3])
        y = np.diff(y)        
        if den == 0:
            err = mean_absolute_error(y[len(actual)-win_actual:len(actual)],actual2[-win_actual:])
            if err > 1:
                return 1.0
            else:
                return err
        else:
            return mean_absolute_error(y[len(actual)-win_actual:len(actual)],actual2[-win_actual:]) / den
    elif model_type == 'exp': 
        x = np.arange(0,365+window_for_averaging,1)
        y = func(x,params[0],params[1])
        if den == 0:
            err = mean_absolute_error(y[0:win_actual],actual2[-win_actual:])
            if err > 1:
                return 1.0
            else:
                return err
        else:
            return mean_absolute_error(y[0:win_actual],actual2[-win_actual:]) / den

    

dates = pd.date_range(start='2020-01-22', end = '2023-06-01')
def get_end_date(p_data):
    _today = datetime.date.today()
    try:
        _end_date = dates[(p_data/p_data.max())>0.999][0].date()
        end_date = _end_date.strftime("%Y-%m-%d")
        days_in_end = (_end_date-_today).days
        if days_in_end<0:
            days_in_end = 0
        return end_date,days_in_end
    except Exception as e:
        return _today,0
def conf_interval(d,z_t="t"):
    std = np.std(d,axis=0)
    n = d.shape[0] # number of windows used
    sq_n = np.sqrt(n)
    z = 1.96 # 95% confidence interval
    m = np.mean(d,axis=0)
    if z_t == "t":
        z_t = stats.t.ppf(1-0.025, df=n)
    else:
        z_t = z
    #return m, m+std, m-std # mean, upper_bound, lower_bound
    return m, m+z_t*(std/sq_n), m-z_t*(std/sq_n) # mean, upper_bound, lower_bound

def get_growth_rate(data,A,lamda):
    #D = region_data["confirmed"]
    #global D
    D = data
    initial_guess = [A,lamda]
    result = optimize.fmin(cost_function,initial_guess,maxfun=1000,maxiter=1000)
    A__,growth_rate__ = result 
    return growth_rate__,A__

def get_unknown_immune_pop(mobility):
    max_possible_immune_pop = 0.95
    imm = 1- ( (max_possible_immune_pop) / (1 + np.exp(-0.05*(mobility))) )
    return imm

def get_shape_trend(data):
    last_val = data[-1]
    max_val = max(data[-14:])
    min_val = min(data[-14:])
    start_val = data[-14]
    #print(data[-14:])
    #print(last_val,min_val,max_val,start_val)
    if last_val  < max_val and start_val  < max_val:
        return "Peak just passed" # need a short window
    elif last_val == min_val:
        return "Decreasing trend" 
    elif last_val == max_val:
        return "increasing trend" # need the upper limit
    elif last_val > min_val:
        return "just started increasing"

###################################################################################################################################


############################################################# Case predictions: ####################################################

def get_dates_to_pred(table_name):
    create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")

    max_date_previous=DataFrame.from_query("""select max(date_of_calc) as max_date_previous from rto.{} where "Country/Region" = 'Zimbabwe'""".format(table_name)).to_pandas()
    max_date_previous=max_date_previous["max_date_previous"].values[0]
    print("max_date_previous",max_date_previous,type(max_date_previous))
    if type(max_date_previous) == str:
        max_date_previous = datetime.datetime.strptime(max_date_previous,"%Y-%m-%d")


    remove_context()

    dates_to_pred= pd.date_range(start=(max_date_previous+datetime.timedelta(days=1)).strftime("%Y-%m-%d"), 
                                 end = MAX_DATE_CASES.strftime("%Y-%m-%d"))#str(np.datetime_as_string(MAX_DATE_CASES,unit='D'))
    return dates_to_pred,max_date_previous

def delete_partial_update(table_name,max_date_previous):
    conection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
    cur = connection.cursor()
    sql_command = """delete from rto.{} where date_of_calc ='{}';""".format(table_name,(max_date_previous+datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
    #print(sql_command)
    cur.execute(sql_command)
    cur.close()

from matplotlib.backends.backend_pdf import PdfPages  
pp = PdfPages('predictions_{}.pdf'.format(str(max_date_cases_tab)))

# Case predictions:
print("Calculating Predictions")

window_for_averaging = 15#14


if len(sys.argv) == 2:
    MAX_DATE_CASES = datetime.datetime.strptime(sys.argv[1],"%Y-%m-%d").date()
else:
    create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
    MAX_DATE_CASES = DataFrame.from_query("""
    select min(max_date) as min_max_date from
    (
    select max("date") as max_date from rto.daily_global 
    union all
    select max("date") as max_date from rto.daily_us
    --union all 
    --select max("date") as max_date from rto.daily_india
    union all 
    select max("date") as max_date from rto.daily_india_districts
    union all
    select max("date") as max_date from rto.stg_daily_global_province
    )as tmp;""").to_pandas().iloc[0,0]
    remove_context()

CURRENT_DAY_SINCE_START = (MAX_DATE_CASES - dates[0].date()).days

dates_to_pred,max_date_previous = get_dates_to_pred("LG_predicted_cases")

#END_DATES_COUNTRIES = pd.DataFrame([],columns=["Country/Region","country_of_state","date_of_calc","pred_end_date","pred_days_remaining_in_epidemic"],dtype=object)
PRED_CASES_COUNTRIES = pd.DataFrame([],columns=["Country/Region","country_of_state","date_of_calc","date","pred_confirmed_cases"],dtype=object)

regions = df_cases[["Country/Region","country_of_state","population","vacc_perc","avg_mob"]].drop_duplicates(subset=["Country/Region","country_of_state"]).sort_values(by="Country/Region")

#create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
#dates_to_pred= pd.date_range(start='2020-10-17', end = '2020-12-02')

#sql_command = """delete from rto.LG_predicted_cases where date_of_calc ='{}';""".format((max_date_previous+datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
#print(sql_command)


def init_params():
    global PEAK_WIN,max_bound_beta,max_bound_alpha,min_bound_alpha,min_bound_beta,min_bound_gamma,max_bound_gamma,min_bound_lambda
    PEAK_WIN = 30
    max_bound_beta = CURRENT_DAY_SINCE_START + PEAK_WIN#1000
    max_bound_alpha = 0.1
    min_bound_alpha = 0.01
    min_bound_beta = 200#CURRENT_DAY_SINCE_START #300 #200
    min_bound_gamma = 1.0
    max_bound_gamma = 1.0
    min_bound_lambda = 0

init_params()

model_params = []
random_runs = 1

WINDOWS = [60]#np.arange(30,60)
FORGET_FACTORS=[0.99,0.9,0.85,0.8]#[0.9,0.95,0.99]#0.999,
MA_WINDOWS = [-1]#[3,7,15]#[2,7,14]

best_scores_all = []
best_scores_all_dict = {}

print(dates_to_pred)
if len(dates_to_pred) > 0:
    # to remove any partial values
    delete_partial_update("LG_predicted_cases",max_date_previous)

    create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
    
    

    prev_params = get_sql_df("""select * from 
                                (select a.*, rank() over(partition by "Country/Region",country_of_state order by best_score_wape asc)  as rank_score
                                from rto.LG_predicted_cases_params a where date_of_calc > (select max(date_of_calc) from rto.LG_predicted_cases_params)-7) tmp
                                where rank_score =1;""")


    for date_to_pred in dates_to_pred:

        max_date_cases = date_to_pred.strftime("%Y-%m-%d")
        #max_date_cases = df_cases["date"].values[-1]
        #print(max_date_cases)
        end_dates_countries = []
        predicted_cases_countries = pd.DataFrame([],columns=["Country/Region","country_of_state","date_of_calc","date","pred_confirmed_cases"],dtype=object)
        
            
        #for country in regions.query("country_of_state == 'US'")["Country/Region"].values:#["United States"]:#countries
        #regions = regions[regions["Country/Region"]  == 'Pune']
        for country,country_of_state,population,vacc_perc,avg_mob in regions.values:
            init_params()
            #sprint(date_to_pred,country,"_",vacc_perc)
            prev_params_item = prev_params[prev_params["Country/Region"]==country].query("country_of_state == '{}'".format(country_of_state))

            #actual_all = df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["confirmed"].values
            actual_all = np.nan_to_num(df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["confirmed"].rolling(window=14).mean().values)  
            #print("---------actual_all",actual_all,len(actual_all))
            actual = copy.copy(actual_all)#[0:i]
            if vacc_perc > 1:
                vacc_perc = 0.1
            
            D=df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["confirmed"].rolling(window=14).mean()
            WINDOW_RISK = 14
            PREDICTIONS = []
            ROLLING_MEAN_FOR_GROWTH_CALC = 0
            gr,A_ = get_growth_rate(D,1000,0.001)
            
            delta_days = CURRENT_DAY_SINCE_START - len(actual) + 1

            max_bound_beta = max_bound_beta - delta_days
            #min_bound_beta = CURRENT_DAY_SINCE_START - delta_days
            print("growth",gr,delta_days,CURRENT_DAY_SINCE_START,min_bound_beta,max_bound_beta)
            if gr > -0.008:
                unknown_immune_pop = get_unknown_immune_pop(avg_mob)
            
                #print(vacc_perc,avg_mob,unknown_immune_pop)
                suscepible_pop = unknown_immune_pop * ( population - (np.max(actual) + 1.3*vacc_perc*population) )
                #suscepible_pop = population - (np.max(actual) + vacc_perc*population)
                #np.max(actual)/max_infected
                '''if gr < 0:
                    min_bound_beta = CURRENT_DAY_SINCE_START - 60 
                   
                else:
                    min_bound_beta = CURRENT_DAY_SINCE_START + 7'''
                    
                
                    
                
                best_score = np.inf
                alpha_best,lamda_best,beta_best=0.02,0,0
                win_best,fg_best =0,0
                #print(prev_params_item,len(prev_params_item))
                if len(prev_params_item)>0 and prev_params_item["beta"].values[0] != -1 and prev_params_item["best_score_wape"].values[0] > 0 and prev_params_item["beta"].values[0] >=min_bound_beta and prev_params_item["beta"].values[0] <= max_bound_beta:
                    prev_alpha = prev_params_item["alpha"].values[0]
                    prev_lamda = prev_params_item["lamda"].values[0]
                    prev_beta = prev_params_item["beta"].values[0]
                    prev_gamma = prev_params_item["gamma"].values[0]
                    best_score = prev_params_item["best_score_wape"].values[0]
                    alpha_best,lamda_best,beta_best,gamma_best = prev_alpha,prev_lamda,prev_beta,prev_gamma
                    if prev_params_item["best_score_wape"].values[0] < 0.1 and prev_params_item["best_score_wape"].values[0] > 0:
                        x0 = [ prev_alpha, prev_lamda, prev_beta, prev_gamma ]
                        random_runs = 1
                    else:
                        #x0 = [0.05,np.max(actual),max_bound_beta-PEAK_WIN/2.,1]
                        random_runs = 3
                        x0 = [np.random.uniform(min_bound_alpha,max_bound_alpha),np.max(actual_all),np.random.uniform(min_bound_beta,max_bound_beta),np.random.uniform(min_bound_gamma,max_bound_gamma)]
                        
                else:
                    #x0 = [0.05,np.max(actual),max_bound_beta-PEAK_WIN/2.,1]
                    random_runs = 3
                    x0 = [np.random.uniform(min_bound_alpha,max_bound_alpha),np.max(actual_all),np.random.uniform(min_bound_beta,max_bound_beta),np.random.uniform(min_bound_gamma,max_bound_gamma)]
                        
                bounds = Bounds([min_bound_alpha,min_bound_lambda,min_bound_beta,min_bound_gamma],[max_bound_alpha, suscepible_pop,max_bound_beta,max_bound_gamma])

                print("x0:",x0,",bounds:",bounds,best_score)
                for ma_win in MA_WINDOWS:
                    for fg in FORGET_FACTORS:
                        forget_factor = fg

                        for win in WINDOWS:
                            WINDOW = win
                            #actual = pd.Series(actual_all).rolling(window=ma_win).mean()#[0:i]
                            #actual = savgol_filter(actual_all, ma_win, 2)
                            #x0 = [0.05,np.max(actual),200]
                            #x0 = [np.random.uniform(0,0.1),np.max(actual_all),np.random.uniform(0,1000)]
                            for j in np.arange(random_runs):
                                if random_runs > 1:
                                    x0 = [np.random.uniform(min_bound_alpha,max_bound_alpha),np.max(actual_all),np.random.uniform(min_bound_beta,max_bound_beta),np.random.uniform(min_bound_gamma,max_bound_gamma)]
                                res = optimize.minimize(fun=cost_predictions,x0=x0,bounds=bounds,method="L-BFGS-B")#method="Nelder-Mead")#,method='Nelder-Mead')
                                alpha,lamda,beta,gamma = res.x
                                current_score = cost_actual([alpha,lamda,beta,gamma])
                                #print(ma_win,win,fg,alpha,lamda,beta,current_score)
                                if current_score < best_score:# and (alpha_best > 0.01 or alpha > 0.01):
                                    best_score = current_score
                                    alpha_best,lamda_best,beta_best,gamma_best = alpha,lamda,beta,gamma
                                    win_best,fg_best = win,fg
                    
                print(date_to_pred,country,win_best,fg_best,ma_win,alpha_best,lamda_best,beta_best,gamma_best,"best_score=",best_score)

            

                model_params.append([country,country_of_state,date_to_pred.strftime("%Y-%m-%d"),alpha_best,lamda_best,beta_best,gamma_best,best_score])
                #end_date = get_end_date(get_predictions_sigmoid(np.arange(0,len(dates),1),alpha_best,lamda_best,beta_best))
                #end_dates_countries.append([country,"",max_date_cases,end_date[0],end_date[1]])
            
                predictions = get_predictions_sigmoid(np.arange(0,len(actual)+365,1)[len(actual)-window_for_averaging:],alpha_best,lamda_best,beta_best,gamma_best)
            else:
                x = np.arange(0,365+window_for_averaging,1)
                predictions = np.cumsum(func(x,A_,gr))
                best_score = cost_actual([A_,gr],"exp")
                model_params.append([country,country_of_state,date_to_pred.strftime("%Y-%m-%d"),gr,A_,-1,-1,best_score])
                print(date_to_pred,country,"A_:",A_,"gr:",gr,"best_score(exp)=",best_score)

            PREDICTIONS.append(predictions)
            PREDICTIONS = np.array(PREDICTIONS)
            pred_mean,pred_up,pred_lower = conf_interval(PREDICTIONS)
            



            # Error correction for total cases
            error_total_cases = df_cases[(df_cases["Country/Region"]==country)&(df_cases["country_of_state"]==country_of_state)]["confirmed"].tail(1).values[0] - pred_mean[window_for_averaging]
            pred_mean = pred_mean + error_total_cases
            
            predictions = pd.DataFrame(pred_mean,columns=["pred_confirmed_cases"])
            predictions["Country/Region"] = country
            predictions["country_of_state"] = country_of_state
            predictions["date_of_calc"] = max_date_cases
            predictions["date"] = (pd.date_range(start=(datetime.datetime.strptime(max_date_cases,"%Y-%m-%d")-datetime.timedelta(days=window_for_averaging)).strftime("%Y-%m-%d"), periods = len(predictions))).strftime("%Y-%m-%d")
            #predictions["date"] = dates[len(actual)-window_for_averaging:len(actual)+365].strftime("%Y-%m-%d")
            #predictions["upper_conf_95"] = pred_up
            #predictions["lower_conf_95"] = pred_lower
            predicted_cases_countries = pd.concat((predicted_cases_countries,predictions),axis=0)

            if country in TD_regions:
                best_scores_all.append(best_score)
                p = predictions
                a = df_cases[df_cases["Country/Region"] ==country].query("country_of_state == '{}' and date <= '{}'".format(country_of_state,max_date_cases))
                #print("---------a ",a,len(a))
                fig = plt.figure(figsize=[10,5])
                param_str = "w={},f={},a={:.4f},l={:.4f},b={:.4f}".format(win_best,fg_best,alpha_best,lamda_best,beta_best)
                plt.title("Cases {},{},{} \nA={:.4f},gr={:.4f}\nx0={}\nbounds={}".format(country,date_to_pred.strftime("%Y-%m-%d"),param_str,
                                                                                                                           A_,gr,
                                                                                                                           ",".join(['{:.4f}'.format(x) for x in x0]),
                                                                                                                           str(bounds)))
                plt.plot(pd.to_datetime(p["date"]),p["pred_confirmed_cases"].diff())
                plt.plot(a["date"],[0]+list(np.diff(actual_all)))
                plt.subplots_adjust(top=0.8)
                pp.savefig(fig)
                #plt.show()
                plt.close()

        #end_dates_countries = pd.DataFrame(end_dates_countries,columns=["Country/Region","country_of_state","date_of_calc","pred_end_date","pred_days_remaining_in_epidemic"])
        
        PRED_CASES_COUNTRIES = pd.concat((PRED_CASES_COUNTRIES,predicted_cases_countries),axis=0)
        #END_DATES_COUNTRIES = pd.concat((END_DATES_COUNTRIES,end_dates_countries),axis=0)
        
        
        #copy_to_sql(df=end_date_all,table_name="LG_predicted_end_dates",schema_name="rto",if_exists="append",primary_index="Country/Region")
        
        copy_to_sql(df=predicted_cases_countries,table_name="LG_predicted_cases",schema_name="rto",if_exists="append",primary_index="Country/Region")
else:
    #create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
    PRED_CASES_COUNTRIES = get_sql_df("""select * from rto.LG_predicted_cases where date_of_calc = (select max(date_of_calc) from rto.LG_predicted_cases);""")
    print("rto.LG_predicted_cases already updated")

if len(model_params) > 0:
    model_params = pd.DataFrame(model_params,columns=["Country/Region","country_of_state","date_of_calc","alpha","lamda","beta","gamma","best_score_wape"])
    copy_to_sql(df=model_params,table_name="LG_predicted_cases_params",schema_name="rto",if_exists="append",primary_index="Country/Region")
    ### save the scores in terms of a histogram and a cdf
    best_scores_all_dict[PEAK_WIN] = best_scores_all
    print("Overrall average WAPE (TD sites):",np.average(best_scores_all))
    _hist,_bin = np.histogram(best_scores_all,bins= np.linspace(0,1.0,21))
    fig = plt.figure()
    plt.title("Avg WAPE ={:.4f}".format(np.average(best_scores_all)))
    plt.plot((100*_bin[1:]).astype(int),np.cumsum(_hist)/np.sum(_hist))
    plt.xlabel("WAPE (%)")
    plt.ylabel("CDF")
    pp.savefig(fig)
    plt.close()

    fig = plt.figure()
    plt.title("Avg WAPE ={:.4f}".format(np.average(best_scores_all)))
    sns.barplot((100*_bin[1:]).astype(int),_hist)
    plt.xlabel("WAPE (%)")
    plt.ylabel("count")
    pp.savefig(fig)
    plt.close()

try:
    remove_context()
except Exception as e:
    pass


PRED_CASES_COUNTRIES["date"] = pd.to_datetime(PRED_CASES_COUNTRIES["date"])
PRED_CASES_COUNTRIES["date_of_calc"] = pd.to_datetime(PRED_CASES_COUNTRIES["date_of_calc"])
PRED_CASES_COUNTRIES




# calculate predictions with confidence
print("Calculating Predictions with confidences")

connection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
cur = connection.cursor()
for date_to_pred in dates_to_pred:
    
    sql_command = """insert into rto.LG_predicted_cases_with_conf
select "Country/Region", country_of_state, date_of_calc,"date", pred_confirmed_cases, upper_conf_95, lower_conf_95 from
(
    select 
    "Country/Region",
    country_of_state,
    "date", 
    max(date_of_calc) as date_of_calc,
    avg(pred_confirmed_cases) as pred_confirmed_cases,
    avg(pred_confirmed_cases)+1.96*stddev_pop(pred_confirmed_cases)/3.472 as upper_conf_95,
    avg(pred_confirmed_cases)-1.96*stddev_pop(pred_confirmed_cases)/3.472 as lower_conf_95
    from 
    (
        select tmp1.date_of_calc,tmp1."date",tmp1."Country/Region",tmp1.country_of_state,tmp1.row_num,tmp2.row_num_total,pred_confirmed_cases from 
        (select date_of_calc,"date","Country/Region",country_of_state,row_number() over(partition by "Country/Region","date" order by "date_of_calc") as row_num,pred_confirmed_cases from  rto.LG_predicted_cases 
        where date_of_calc <= '{}' --"Country/Region" in ('United States','Germany') 
        ) tmp1
        inner join 
        (select "date","Country/Region",country_of_state,count(*) as row_num_total from  rto.LG_predicted_cases 
        where date_of_calc <= '{}' --where "Country/Region"in ('United States','Germany') 
        group by 1,2,3) tmp2
        on tmp1."Country/Region" = tmp2."Country/Region" and tmp1."date" = tmp2."date" and tmp1.country_of_state=tmp2.country_of_state
        --order by tmp1."date",tmp1.date_of_calc
        where tmp2.row_num_total - tmp1.row_num <14 and row_num_total>=14
    ) as tmp
    group by 1,2,3
) as tmp_main
where "date"> (select max(date_of_calc) from rto.LG_predicted_cases) 
and date_of_calc not in (select distinct date_of_calc from rto.LG_predicted_cases_with_conf);""".format(date_to_pred.strftime("%Y-%m-%d"),date_to_pred.strftime("%Y-%m-%d"))
    #print(sql_command)
    cur.execute(sql_command)
    r = cur.fetchall()
    print(date_to_pred)
cur.close()



###################################################################################################################################
logger.info("rto.LG_predicted_cases, rto.LG_predicted_cases_with_conf done. Predictions (cases) done.")

############################################### Case predictions (Deaths): #########################################################

# Case predictions (Deaths):
print("Calculating Predictions (DEATHS)")

window_for_averaging = 15#14

PRED_DEATHS_COUNTRIES = pd.DataFrame([],columns=["Country/Region","country_of_state","date_of_calc","date","pred_deaths"],dtype=object)

regions = df_cases[["Country/Region","country_of_state","population"]].drop_duplicates(subset=["Country/Region","country_of_state"]).sort_values(by="Country/Region")

#create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")

dates_to_pred,max_date_previous = get_dates_to_pred("LG_predicted_deaths")

print(dates_to_pred)


model_params = []
random_runs = 1
best_scores_all_d = []
best_scores_all_d_dict = {}
#dates_to_pred= pd.date_range(start='2020-10-17', end = '2020-11-29')
if len(dates_to_pred)>0:
    prev_params = get_sql_df("""select * from 
                                (select a.*, rank() over(partition by "Country/Region",country_of_state order by best_score_wape asc)  as rank_score
                                from rto.LG_predicted_deaths_params a where date_of_calc > (select max(date_of_calc) from rto.LG_predicted_deaths_params)-7) tmp
                                where rank_score =1;""")

    # to remove any partial values
    delete_partial_update("LG_predicted_deaths",max_date_previous)
    
    create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
    for date_to_pred in dates_to_pred:
        max_date_cases = date_to_pred.strftime("%Y-%m-%d")
        #max_date_cases = df_cases["date"].values[-1]
        print(max_date_cases)
        end_dates_countries = []
        predicted_deaths_countries = pd.DataFrame([],columns=["Country/Region","country_of_state","date_of_calc","date","pred_deaths"],dtype=object)
        #WINDOWS = [60]#np.arange(30,60)
        #FORGET_FACTORS=[0.99,0.9,0.85]#[0.9,0.95,0.99]
        #MA_WINDOWS = [3,5]#7,15]#[2,7,14]
            
        #for country in regions.query("country_of_state == 'US'")["Country/Region"].values:#["United States"]:#countries
        for country,country_of_state,population in regions.values:
            print(date_to_pred,country)
            init_params()
            prev_params_item = prev_params[prev_params["Country/Region"]==country].query("country_of_state == '{}'".format(country_of_state))

            #actual_all = df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["deaths"].values
            actual_all = np.nan_to_num(df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["deaths"].rolling(window=14).mean().values)
        
            #actual_all = np.nan_to_num(df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["deaths"].rolling(window=14).mean().values)
            actual = copy.copy(actual_all)

            D=df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["deaths"].rolling(window=14).mean()
            WINDOW_RISK = 14
            PREDICTIONS = []
            ROLLING_MEAN_FOR_GROWTH_CALC = 0
            gr_d,A_d = get_growth_rate(D,1000,0.001)

            # for setting the max bound for deaths based on case predictions
            cases = df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["confirmed"].values
            mortality_rate = actual_all[-1]/cases[-1]
            pred_cases = PRED_CASES_COUNTRIES[PRED_CASES_COUNTRIES["Country/Region"]==country].query("country_of_state == '"+country_of_state+"'")["pred_confirmed_cases"].values
            #- actual_all[-1]

            
            delta_days = CURRENT_DAY_SINCE_START - len(actual) + 1
            max_bound_beta = max_bound_beta - delta_days
            #min_bound_beta = CURRENT_DAY_SINCE_START - delta_days

            PREDICTIONS_DEATHS = []
            print("growth_deaths",gr_d,delta_days,CURRENT_DAY_SINCE_START,min_bound_beta,max_bound_beta)
            if gr_d > -0.008:
                best_score = np.inf
                alpha_best,lamda_best,beta_best=0.02,0,0
                win_best,fg_best =0,0

                '''if gr_d < 0:
                    min_bound_beta = CURRENT_DAY_SINCE_START - 60
                else:
                    min_bound_beta = CURRENT_DAY_SINCE_START + 7'''

                if len(prev_params_item)>0 and prev_params_item["beta"].values[0] != -1 and prev_params_item["best_score_wape"].values[0] > 0 and prev_params_item["beta"].values[0] >=min_bound_beta and prev_params_item["beta"].values[0] <= max_bound_beta:
                    prev_alpha = prev_params_item["alpha"].values[0]
                    prev_lamda = prev_params_item["lamda"].values[0]
                    prev_beta = prev_params_item["beta"].values[0]
                    prev_gamma = prev_params_item["gamma"].values[0]
                    best_score = prev_params_item["best_score_wape"].values[0]
                    alpha_best,lamda_best,beta_best,gamma_best = prev_alpha,prev_lamda,prev_beta,prev_gamma
                    if prev_params_item["best_score_wape"].values[0] < 0.1 and prev_params_item["best_score_wape"].values[0] > 0:
                        x0 = [ prev_alpha, prev_lamda, prev_beta, prev_gamma ]
                        random_runs = 1
                    else:
                        #x0 = [0.05,np.max(actual),max_bound_beta-PEAK_WIN/2.,1]
                        random_runs = 3
                        x0 = [np.random.uniform(min_bound_alpha,max_bound_alpha),np.max(actual_all),np.random.uniform(min_bound_beta,max_bound_beta),np.random.uniform(min_bound_gamma,max_bound_gamma)]
                        
                else:
                    #x0 = [0.05,np.max(actual),max_bound_beta-PEAK_WIN/2.,1]
                    random_runs = 3
                    x0 = [np.random.uniform(min_bound_alpha,max_bound_alpha),np.max(actual_all),np.random.uniform(min_bound_beta,max_bound_beta),np.random.uniform(min_bound_gamma,max_bound_gamma)]
                
                bounds = Bounds([min_bound_alpha, min_bound_lambda,min_bound_beta,min_bound_gamma], [max_bound_alpha, mortality_rate*(pred_cases[-1] - pred_cases[window_for_averaging])  ,max_bound_beta+15,max_bound_gamma])#np.max(actual)/max_infected
                
                print("x0:",x0,",bounds:",bounds,best_score)
                
                for ma_win in MA_WINDOWS:
                    for fg in FORGET_FACTORS:
                        forget_factor = fg

                        for win in WINDOWS:
                            WINDOW = win
                            #actual = pd.Series(actual_all).rolling(window=ma_win).mean()#[0:i]
                            #actual = savgol_filter(actual_all, ma_win, 2)
                            #x0 = [0.05,actual_all[-1],200]
                            #x0 = [np.random.uniform(0,0.1),np.max(actual_all),np.random.uniform(0,1000)]
                            for j in np.arange(random_runs):
                                if random_runs > 1:
                                    x0 = [np.random.uniform(min_bound_alpha,max_bound_alpha),np.max(actual_all),np.random.uniform(min_bound_beta,max_bound_beta),np.random.uniform(min_bound_gamma,max_bound_gamma)]
                                res = optimize.minimize(fun=cost_predictions,x0=x0,bounds=bounds,method="L-BFGS-B")
                                alpha,lamda,beta,gamma = res.x
                                current_score = cost_actual([alpha,lamda,beta,gamma])
                                print(ma_win,win,fg,alpha,lamda,beta,gamma,current_score)
                                if current_score < best_score:# and (alpha_best > 0.01 or alpha > 0.01):
                                    best_score = current_score
                                    alpha_best,lamda_best,beta_best,gamma_best = alpha,lamda,beta,gamma
                                    win_best,fg_best = win,fg
                            
                print(date_to_pred,country,win_best,fg_best,ma_win,alpha_best,lamda_best,beta_best,gamma_best,"best_score=",best_score) 
                model_params.append([country,country_of_state,date_to_pred.strftime("%Y-%m-%d"),alpha_best,lamda_best,beta_best,gamma_best,best_score])

                #end_date = get_end_date(get_predictions_sigmoid(np.arange(0,len(dates),1),alpha_best,lamda_best,beta_best,gamma_best))
                #end_dates_countries.append([country,"",max_date_cases,end_date[0],end_date[1]])
                predictions_deaths = get_predictions_sigmoid(np.arange(0,len(actual)+365,1)[len(actual)-window_for_averaging:],alpha_best,lamda_best,beta_best,gamma_best)
            else:
                x = np.arange(0,365+window_for_averaging,1)
                predictions_deaths = np.cumsum(func(x,A_d,gr_d))
                best_score = cost_actual([A_d,gr_d],"exp")
                model_params.append([country,country_of_state,date_to_pred.strftime("%Y-%m-%d"),gr_d,A_d,-1,-1,best_score])
                print(date_to_pred,country,"A_d:",A_d,"gr_d:",gr_d,"best_score(exp)=",best_score)

            PREDICTIONS_DEATHS.append(predictions_deaths)
                    
            PREDICTIONS_DEATHS = np.array(PREDICTIONS_DEATHS)
            pred_mean,pred_up,pred_lower = conf_interval(PREDICTIONS_DEATHS)
            
            # Error correction in total deaths
            error_total_deaths = df_cases[(df_cases["Country/Region"]==country)&(df_cases["country_of_state"]==country_of_state)]["deaths"].tail(1).values[0] - pred_mean[window_for_averaging]
            pred_mean = pred_mean + error_total_deaths

            '''
            # for death prediction correction
            cases = df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["confirmed"].values
            mortality_rate = actual_all[-1]/cases[-1]
            #print(mortality_rate,"mor")
            #print(len(pred_mean),len(correcion_factor_deaths))
            pred_cases = PRED_CASES_COUNTRIES[PRED_CASES_COUNTRIES["Country/Region"]==country].query("country_of_state == '"+country_of_state+"'")["pred_confirmed_cases"].values
            correcion_factor_deaths = 0.5
            corrected_deaths = correcion_factor_deaths*pred_mean + (1-correcion_factor_deaths)*pred_cases*mortality_rate
            
            _start_total_death_pred = pred_mean[0]
            corrected_deaths = np.diff(corrected_deaths)
            corrected_deaths[corrected_deaths<0] = 0

            corrected_deaths = [pred_mean[0]] + list(corrected_deaths)
            corrected_deaths = np.cumsum(corrected_deaths)'''

            
            predictions_deaths = pd.DataFrame(pred_mean,columns=["pred_deaths"])
            predictions_deaths["Country/Region"] = country
            predictions_deaths["country_of_state"] = country_of_state
            predictions_deaths["date_of_calc"] = max_date_cases
            predictions_deaths["date"] = (pd.date_range(start=(datetime.datetime.strptime(max_date_cases,"%Y-%m-%d")-datetime.timedelta(days=window_for_averaging)).strftime("%Y-%m-%d"), periods = len(predictions_deaths))).strftime("%Y-%m-%d")
            #predictions_deaths["date"] = dates[len(actual)-window_for_averaging:len(actual)+365].strftime("%Y-%m-%d")
            #predictions_deaths["upper_conf_95"] = pred_up
            #predictions_deaths["lower_conf_95"] = pred_lower
            predicted_deaths_countries = pd.concat((predicted_deaths_countries,predictions_deaths),axis=0)

            
            if country in TD_regions:
                best_scores_all_d.append(best_score)
                p = predictions_deaths
                a = df_cases[df_cases["Country/Region"] ==country].query("country_of_state == '{}' and date <= '{}'".format(country_of_state,max_date_cases))
                plt.plot(p["pred_deaths"])
                fig = plt.figure(figsize=[10,5])
                param_str = "w={},f={},a={:.4f},l={:.4f},b={:.4f}".format(win_best,fg_best,alpha_best,lamda_best,beta_best)
                plt.title("Deaths {},{},{} \nA={:.4f},gr={:.4f},mort_cur={:.4f},mort_pred={:.4f}\nx0={}\nbounds={}".format(country,date_to_pred.strftime("%Y-%m-%d"),param_str,
                                                                                                                           A_d,gr_d,
                                                                                                                           mortality_rate,pred_mean[-1]/pred_cases[-1],
                                                                                                                           ",".join(['{:.4f}'.format(x) for x in x0]),
                                                                                                                           str(bounds)))
                plt.plot(pd.to_datetime(p["date"]),p["pred_deaths"].diff())
                plt.plot(a["date"],[0]+list(np.diff(actual_all)))
                plt.subplots_adjust(top=0.8)
                pp.savefig(fig)
                plt.close()
                #plt.show()

        #end_dates_countries = pd.DataFrame(end_dates_countries,columns=["Country/Region","country_of_state","date_of_calc","pred_end_date","pred_days_remaining_in_epidemic"])
        
        PRED_DEATHS_COUNTRIES = pd.concat((PRED_DEATHS_COUNTRIES,predicted_deaths_countries),axis=0)
            
        
        copy_to_sql(df=predicted_deaths_countries,table_name="LG_predicted_deaths",schema_name="rto",if_exists="append",primary_index="Country/Region")
else:
    create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
    print("rto.LG_predicted_deaths already updated")

if len(model_params) > 0:
    model_params = pd.DataFrame(model_params,columns=["Country/Region","country_of_state","date_of_calc","alpha","lamda","beta","gamma","best_score_wape"])
    copy_to_sql(df=model_params,table_name="LG_predicted_deaths_params",schema_name="rto",if_exists="append",primary_index="Country/Region")
    print("Overrall average WAPE (TD sites) [DEATHS]:",np.average(best_scores_all_d))
    _hist,_bin = np.histogram(best_scores_all_d,bins= np.linspace(0,1.0,21))
    fig = plt.figure()
    plt.title("Avg WAPE ={:.4f}".format(np.average(best_scores_all_d)))
    plt.plot((100*_bin[1:]).astype(int),np.cumsum(_hist)/np.sum(_hist))
    plt.xlabel("WAPE (%)")
    plt.ylabel("CDF")
    pp.savefig(fig)
    plt.close()

    fig = plt.figure()
    plt.title("Avg WAPE ={:.4f}".format(np.average(best_scores_all_d)))
    sns.barplot((100*_bin[1:]).astype(int),_hist)
    plt.xlabel("WAPE (%)")
    plt.ylabel("count")
    pp.savefig(fig)
    plt.close()

remove_context()

pp.close()


PRED_DEATHS_COUNTRIES["date"] = pd.to_datetime(PRED_DEATHS_COUNTRIES["date"])
PRED_DEATHS_COUNTRIES["date_of_calc"] = pd.to_datetime(PRED_DEATHS_COUNTRIES["date_of_calc"])
PRED_DEATHS_COUNTRIES





###################################################################################################################################
logger.info("rto.LG_predicted_deaths updated. Predictions (deaths) done.")

############################################### TD Site Analysis Dashboard #########################################################


import teradatasql
connection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
cur = connection.cursor()
sql_command = """
select aaa.*,ddd.country_of_state,ddd."date",ddd.pred_confirmed_cases as confirmed,ddd.pred_deaths as deaths from 
(
    select 
    distinct b.county_district as "Country/Region"
    ,popTotal as population 
    --,a.pred_confirmed_cases as confirmed
    
    from rto.population_consolidated_vw a
    inner join
    (select * from rto.td_sites where country_region='US') b
    on a.county_district = b.county_district and a.province_state = b.state_province
    
    
    union all
    
    select distinct country_region as "Country/Region",pop_country as population from rto.country_population aa
    inner join
    (select * from rto.td_sites where country_region<>'US' and country_region <>'India' and country_region <> 'Australia' and country_region <> 'Japan') bb
    on aa.country = bb.country_region
    
    union all 
    
    select distinct state as "Country/Region",population from rto.population_india_states aa
    inner join
    (select * from rto.td_sites where country_region ='India') bb
    on aa.state = bb.state_province
    where state = 'Telangana'

    union all 

    select distinct district as "Country/Region",population from rto.population_india_districts aa
    inner join
    (select * from rto.td_sites where country_region ='India') bb
    on aa.state = bb.state_province
    where district <>  'Hyderabad'
    
    union all
    
    select distinct location as "Country/Region",popTotal from rto.population_global aa
    inner join
    (select * from rto.td_sites where country_region in ('Australia','Japan')) bb
    on "Country/Region" = bb.state_province
    

) aaa

inner join
(
select bbb."Country/Region",bbb.country_of_state,bbb."date",bbb.pred_confirmed_cases,ccc.pred_deaths from 
(select * from rto.LG_predicted_cases where country_of_state in ('','US_county','India','India_district','Australia','Japan') and date_of_calc = (select max(date_of_calc) from rto.LG_predicted_cases)) bbb
inner join
(select * from rto.LG_predicted_deaths where country_of_state in ('','US_county','India','India_district','Australia','Japan') and date_of_calc = (select max(date_of_calc) from rto.LG_predicted_deaths)) ccc
on bbb."date" = ccc."date" and bbb."Country/Region" = ccc."Country/Region" and bbb.country_of_state = ccc.country_of_state
) ddd
on aaa."Country/Region" = ddd."Country/Region"
order by "date" ;
"""
#print(sql_command)
cur.execute(sql_command)
res = cur.fetchall()
cur.close()

df_cases_pred = pd.DataFrame(res,columns=np.array(cur.description)[:,0])
df_cases_pred["date"] = pd.to_datetime(df_cases_pred["date"])
df_cases_pred = df_cases_pred.fillna(value=df_cases_pred["population"].mean())
df_cases_pred = df_cases_pred.fillna(value=df_cases_pred["population"].mean())
pop = df_cases_pred["population"].values
pop[pop==0]= df_cases_pred["population"].mean()
df_cases_pred["population"] = pop
min_date_for_future_risk = df_cases_pred["date"].min()+datetime.timedelta(days=window_for_averaging)
df_cases_pred



dates_to_calc = [datetime.datetime.strptime(min_date_for_future_risk.strftime("%Y-%m-%d"),"%Y-%m-%d") + datetime.timedelta(days=30),
 datetime.datetime.strptime(min_date_for_future_risk.strftime("%Y-%m-%d"),"%Y-%m-%d") + datetime.timedelta(days=60),
 datetime.datetime.strptime(min_date_for_future_risk.strftime("%Y-%m-%d"),"%Y-%m-%d") + datetime.timedelta(days=90)]
dates_to_calc

###################################################################################################################################



############################################### calculate risk on predicted data #########################################################

# to check if the table is already updated or not.
connection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
cur = connection.cursor()
sql_command = """select max(date_of_calc) from rto.regional_intensity_profiling_future where "Country/Region" = 'Wake'"""
cur.execute(sql_command)
res = cur.fetchall()
cur.close()
max_date_fut_tab = res[0][0]
#


# calculate risk on predicted data

ROLLING_MEAN_FOR_GROWTH_CALC = 0
REGIONS = pd.DataFrame([],columns=['Country/Region', 'country_of_state', 'population', 'date', 'is_TD',
       'growth_rate', 'growth_rate_deaths', 'Re', 'total_cases',
       'total_cases_per_M', 'daily_cases', 'daily_cases_per_M', 'daily_deaths',
       'daily_deaths_per_M'],dtype=object)
for date_to_calc in dates_to_calc:
    total_cases = []
    daily_cases = []
    daily_deaths = []
    growth_rates = []
    growth_rates_deaths = []
    WINDOW_RISK = 14
    A = 1000
    lamda = 0.001

    regions = df_cases_pred[["Country/Region","country_of_state","population"]].drop_duplicates(subset=["Country/Region","country_of_state"]).sort_values(by="Country/Region")
    max_date_cases = df_cases_pred["date"].values[-1]
    regions["date"] = date_to_calc.date().strftime("%Y-%m-%d")#max_date_cases
    IS_TD = []
    print(date_to_calc)
    for row in regions[["Country/Region","country_of_state","population"]].values:

        region,country_of_state,pop = row
        #print(region,country_of_state,pop)
        if region in TD_regions:
            IS_TD.append(1)
        else:
            IS_TD.append(0)

        #region_data = df_cases_pred[df_cases_pred["Country/Region"]== region].query("country_of_state == '"+str(country_of_state)+"'")
        region_data = df_cases_pred[df_cases_pred["Country/Region"]== region].query("country_of_state == '"+str(country_of_state)+"' and date <= '"+date_to_calc.date().strftime("%Y-%m-%d")+"'")
       
        D = region_data["confirmed"]
        #print(D)
        initial_guess = [A,lamda]
        result = optimize.fmin(cost_function,initial_guess,maxfun=1000,maxiter=1000)
        A_,growth_rate = result 
        growth_rates.append(growth_rate)
        print(region,growth_rate)


        D = region_data["deaths"]#D = COUNTRY_DATA[country]["JHU_ConfirmedDeaths.data"]
        initial_guess = [A,lamda]
        result = optimize.fmin(cost_function,initial_guess,maxfun=1000,maxiter=1000)
        A_,growth_rate = result 
        growth_rates_deaths.append(growth_rate)

        total_cases.append(float(region_data["confirmed"].values[-1]))
        daily_cases.append(region_data["confirmed"].diff().values[-WINDOW_RISK:].mean())
        daily_deaths.append(region_data["deaths"].diff().values[-WINDOW_RISK:].mean())
    growth_rates = np.array(growth_rates)
    growth_rates_deaths = np.array(growth_rates_deaths)
    growth_rates[growth_rates<-0.5] = -0.5 # clip the negative value
    growth_rates_deaths[growth_rates_deaths<-0.5] = -0.5 # clip the negative value
    regions["is_TD"] = IS_TD
    regions["growth_rate"] = growth_rates
    regions["growth_rate_deaths"] = growth_rates_deaths
    Re = 1 + growth_rates*5
    Re[Re<0] = 0
    regions["Re"] = Re
    regions["total_cases"] = total_cases
    regions["total_cases_per_M"] = 1000000*regions["total_cases"]/regions["population"]
    daily_cases = np.array(daily_cases)
    daily_cases[daily_cases<0]=0
    daily_deaths = np.array(daily_deaths)
    daily_deaths[daily_deaths<0]=0
    regions["daily_cases"] = daily_cases
    regions["daily_cases_per_M"] = 1000000*regions["daily_cases"]/regions["population"]
    regions["daily_deaths"] = daily_deaths
    regions["daily_deaths_per_M"] = 1000000*regions["daily_deaths"]/regions["population"]
    REGIONS = pd.concat((REGIONS,regions),axis=0)
#REGIONS = REGIONS.dropna()
#create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
#copy_to_sql(df=REGIONS,table_name="regional_intensity_profiling",schema_name="rto",if_exists="append",primary_index="Country/Region")
#remove_context()



countries = REGIONS[["Country/Region","country_of_state","is_TD","population"]].drop_duplicates()
cols = ["Country/Region",
        "country_of_state",
        "population",
        "date_of_calc",
        "is_TD",
        "daily_cases_per_M_30",
       "daily_cases_per_M_60",
       "daily_cases_per_M_90",
       "daily_deaths_per_M_30",
       "daily_deaths_per_M_60",
       "daily_deaths_per_M_90",
       "Re_30",
        "Re_60",
        "Re_90",
        "growth_rate_30",
        "growth_rate_60",
        "growth_rate_90",
        "growth_rate_deaths_30",
        "growth_rate_deaths_60",
        "growth_rate_deaths_90"
       ]
REGIONS_PIVOTED = []
for country,country_of_state,is_TD,population in countries.values:
#print(country,",",country_of_state)
    reg_country = REGIONS[(REGIONS["Country/Region"]==country) & (REGIONS["country_of_state"] == country_of_state)]
    #print(reg_country)
    row=[country,
       country_of_state,
       population,
        min_date_for_future_risk.strftime("%Y-%m-%d"),
       #max_date_cases.strftime("%Y-%m-%d"),
       is_TD]+\
    list(reg_country["daily_cases_per_M"].values.T)\
    +list(reg_country["daily_deaths_per_M"].values.T)\
    +list(reg_country["Re"].values.T)\
    +list(reg_country["growth_rate"].values.T)\
    +list(reg_country["growth_rate_deaths"].values.T)
    REGIONS_PIVOTED.append(row)
    #print(row)
REGIONS_PIVOTED = pd.DataFrame(REGIONS_PIVOTED,columns=cols)
REGIONS_PIVOTED



a=min_date_for_future_risk.strftime("%Y-%m-%d")
b=(min_date_for_future_risk - datetime.timedelta(days=14)).strftime("%Y-%m-%d")
c=(min_date_for_future_risk - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
d=(min_date_for_future_risk - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
a,b,c,d



create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
REGIONS_CURRENT = DataFrame.from_query("""select * from rto.regional_intensity_profiling where "date" in('{}','{}','{}','{}');""".format(a,b,c,d)).to_pandas()
remove_context()
REGIONS_CURRENT= REGIONS_CURRENT.reset_index()
REGIONS_CURRENT = REGIONS_CURRENT.sort_values("date",ascending=False)
REGIONS_CURRENT 


countries = REGIONS[["Country/Region","country_of_state","is_TD","population"]].drop_duplicates()
cols = ["Country/Region",
        "country_of_state",
        "population",
        "date_of_calc",
        "is_TD",
        "daily_cases_per_M",
        "daily_cases_per_M_14P",
       "daily_cases_per_M_30P",
       "daily_cases_per_M_60P",
       "daily_deaths_per_M",
       "daily_deaths_per_M_14P",
       "daily_deaths_per_M_30P",
       "daily_deaths_per_M_60P",
       "Re",
       "Re_14P",
        "Re_30P",
        "Re_60P",
        "growth_rate",
        "growth_rate_14P",
        "growth_rate_30P",
        "growth_rate_60P",
        "growth_rate_deaths",
        "growth_rate_deaths_14P",
        "growth_rate_deaths_30P",
        "growth_rate_deaths_60P"
       ]
REGIONS_PIVOTED2 = []
for country,country_of_state,is_TD,population in countries.values:
#print(country,",",country_of_state)
    reg_country = REGIONS_CURRENT[(REGIONS_CURRENT["Country/Region"]==country) & (REGIONS_CURRENT["country_of_state"] == country_of_state)]
    #print(reg_country)
    _append_empty = [] # in case of no historical values
    for i in range(4-reg_country.shape[0]):
        _append_empty.append(None)
    #print(_append_empty)
    
    row=[country,
       country_of_state,
       population,
        min_date_for_future_risk.strftime("%Y-%m-%d"),
       #max_date_cases.strftime("%Y-%m-%d"),
       is_TD]+\
    list(reg_country["daily_cases_per_M"].values.T) + _append_empty\
    +list(reg_country["daily_deaths_per_M"].values.T) + _append_empty\
    +list(reg_country["Re"].values.T) + _append_empty\
    +list(reg_country["growth_rate"].values.T) + _append_empty\
    +list(reg_country["growth_rate_deaths"].values.T) + _append_empty
    REGIONS_PIVOTED2.append(row)
    #print(row)
REGIONS_PIVOTED2 = pd.DataFrame(REGIONS_PIVOTED2,columns=cols)
REGIONS_PIVOTED2



REGIONS_MERGED = REGIONS_PIVOTED2.merge(REGIONS_PIVOTED,on=["Country/Region","country_of_state"])
del REGIONS_MERGED["is_TD_y"]
del REGIONS_MERGED["population_y"]
del REGIONS_MERGED["date_of_calc_y"]
REGIONS_MERGED = REGIONS_MERGED.rename(columns = {"is_TD_x":"is_TD","population_x":"population","date_of_calc_x":"date_of_calc"})

if max_date_fut_tab != min_date_for_future_risk.strftime("%Y-%m-%d"):
    create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
    copy_to_sql(df=REGIONS_MERGED,table_name="regional_intensity_profiling_future",schema_name="rto",if_exists="append",primary_index="Country/Region")
    remove_context()


    logger.info("rto.regional_intensity_profiling_future done. Risk prediction for specific future dates.")
    print("rto.regional_intensity_profiling_future updated")
else:
    print("rto.regional_intensity_profiling_future already updated")

###################################################################################################################################



################################################# risk predictions for all countries #############################################

# to check if the table is already updated or not.
connection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
cur = connection.cursor()
sql_command = """select max(date_of_calc) from rto.regional_intensity_predictions where "Country/Region" = 'Wake'"""
cur.execute(sql_command)
res = cur.fetchall()
cur.close()
max_date_risk_pred_tab = res[0][0]
#

print("risk predictions (365 days into the future)")
#df_cases_pred2 = df_cases[df_cases["Country/Region"].isin(df_cases_pred["Country/Region"].unique())]
#df_cases_pred2 = pd.concat((df_cases_pred2,df_cases_pred))

if max_date_risk_pred_tab != MAX_DATE_CASES.strftime("%Y-%m-%d"):

    df_cases_pred2 = df_cases_pred
    dates_to_calc= pd.date_range(start=(MAX_DATE_CASES+datetime.timedelta(days=0)).strftime("%Y-%m-%d"), 
                                 end = (MAX_DATE_CASES+datetime.timedelta(days=365)).strftime("%Y-%m-%d"))#str(np.datetime_as_string(MAX_DATE_CASES,unit='D'))
    dates_to_calc


    region_data_hist = REGIONS_CURRENT[REGIONS_CURRENT["Country/Region"]== region].query("country_of_state == '"+str(country_of_state)+"' and date == '"+date_to_calc.date().strftime("%Y-%m-%d")+"'")


    # calculate risk on predicted data 365 days into the future
    ROLLING_MEAN_FOR_GROWTH_CALC = 0
    REGIONS = pd.DataFrame([],columns=['Country/Region', 'country_of_state', 'population', 'date', 'is_TD',
           'growth_rate', 'growth_rate_deaths', 'Re', 'total_cases',
           'total_cases_per_M', 'daily_cases', 'daily_cases_per_M', 'daily_deaths',
           'daily_deaths_per_M'],dtype=object)

    error_forget_factor = 0.99
    power_error_factor = 0
    errors_total_cases = [] # errors for each country
    errors_daily_cases = []
    errors_daily_deaths = []
    errors_growth_rates = []
    errors_growth_rates_deaths = []
    errors_Re = []

    error_total_cases = 0
    error_daily_cases = 0
    error_daily_deaths = 0
    error_growth_rates = 0
    error_growth_rates_deaths = 0
    error_Re = 0

    day_risk = 0
    for date_to_calc in dates_to_calc:
        total_cases = []
        daily_cases = []
        daily_deaths = []
        growth_rates = []
        growth_rates_deaths = []
        
        
        
        
        is_pred = []
        WINDOW_RISK = 14
        A = 1000
        lamda = 0.001

        regions = df_cases_pred2[["Country/Region","country_of_state","population"]].drop_duplicates(subset=["Country/Region","country_of_state"]).sort_values(by="Country/Region")
        max_date_cases = df_cases_pred2["date"].values[-1]
        regions["date"] = date_to_calc.date().strftime("%Y-%m-%d")#max_date_cases
        IS_TD = []
        #print(date_to_calc)
        if date_to_calc.date() <= MAX_DATE_CASES:
            is_pred = "Actual" 
        else:
            is_pred = "Predicted"
            
        #regions = regions[(regions["Country/Region"] =='Japan') | (regions["Country/Region"] =='Pakistan') |(regions["Country/Region"]=='United Arab Emirates')]
        for row in regions[["Country/Region","country_of_state","population"]].values:

            region,country_of_state,pop = row
            #print(region,country_of_state,pop)
            if region in TD_regions:
                IS_TD.append(1)
            else:
                IS_TD.append(0)
                
            

            #region_data = df_cases_pred2[df_cases_pred2["Country/Region"]== region].query("country_of_state == '"+str(country_of_state)+"'")
            region_data = df_cases_pred2[df_cases_pred2["Country/Region"]== region].query("country_of_state == '"+str(country_of_state)+"' and date <= '"+date_to_calc.date().strftime("%Y-%m-%d")+"'")
            
            
            
            
            if day_risk % 7 == 0:
                D = region_data["confirmed"]
                initial_guess = [A,lamda]
                result = optimize.fmin(cost_function,initial_guess,maxfun=1000,maxiter=1000,disp = False)
                A_,growth_rate_c = result 


                D = region_data["deaths"]#D = COUNTRY_DATA[country]["JHU_ConfirmedDeaths.data"]
                initial_guess = [A,lamda]
                result = optimize.fmin(cost_function,initial_guess,maxfun=1000,maxiter=1000,disp = False)
                A_,growth_rate_d = result 
                if day_risk == 0:
                    print(region,growth_rate_c)
                

            else:
                growth_rate_c = np.NaN
                growth_rate_d = np.NaN

            #print(date_of_calc,region,growth_rate_c)


         
            ####### for error term to correct the predictions
            if date_to_calc.date() == MAX_DATE_CASES:
                ef = [error_forget_factor**i for i in range(len(region_data))]
                region_data_hist = REGIONS_CURRENT[REGIONS_CURRENT["Country/Region"]== region].query("country_of_state == '"+str(country_of_state)+"' and date == '"+date_to_calc.date().strftime("%Y-%m-%d")+"'")
                
                ## old method
                '''error_total_cases = float(region_data_hist["total_cases"].values[-1]) - float(region_data["confirmed"].values[-1])
                error_daily_cases = region_data_hist["daily_cases"].values[-1] - region_data["confirmed"].diff().values[-WINDOW:].mean() 
                error_daily_deaths = region_data_hist["daily_deaths"].values[-1] - region_data["deaths"].diff().values[-WINDOW:].mean()
                error_growth_rates = region_data_hist["growth_rate"].values[-1] - growth_rate_c 
                error_growth_rates_deaths = region_data_hist["growth_rate_deaths"].values[-1] - growth_rate_d
                error_Re = region_data_hist["Re"].values[-1] - ( 1 + growth_rate_c*5 )'''

                # new method
                error_total_cases = float(region_data_hist["total_cases"].values[-1])/float(region_data["confirmed"].values[-1])
                error_daily_cases = region_data_hist["daily_cases"].values[-1]/ (0.001+ region_data["confirmed"].diff().values[-WINDOW_RISK:].mean() )
                error_daily_deaths = region_data_hist["daily_deaths"].values[-1]/( 0.001 + region_data["deaths"].diff().values[-WINDOW_RISK:].mean() )
                error_growth_rates = region_data_hist["growth_rate"].values[-1]/ (0.00001 + growth_rate_c )
                error_growth_rates_deaths = region_data_hist["growth_rate_deaths"].values[-1]/ ( 0.00001 + growth_rate_d)
                error_Re = region_data_hist["Re"].values[-1]/(0.00001 + 1 + growth_rate_c*5 )


                                
                #print(error_total_cases,error_daily_cases,)
                '''assert type(error_total_cases)==float or type(error_total_cases)==int 
                assert type(error_daily_cases)==float or type(error_daily_cases)==int
                assert type(error_daily_deaths)==float or type(error_daily_deaths)==int
                assert type(error_growth_rates)==float or type(error_growth_rates)==int
                assert type(error_growth_rates_deaths)==float or  type(error_growth_rates_deaths)==int 
                assert type(error_Re)==float or type(error_Re)==int'''
                
            
            
                errors_total_cases.append(error_total_cases)
                errors_daily_cases.append(error_daily_cases)
                errors_daily_deaths.append(error_daily_deaths)
                errors_growth_rates.append(error_growth_rates)
                errors_growth_rates_deaths.append(error_growth_rates_deaths)
                errors_Re.append(error_Re)

                '''if region == 'Pune':
                    print(date_to_calc)
                    print(region)
                    print(error_daily_deaths)
                    print(region_data["deaths"].diff().values[-WINDOW_RISK:].mean() )
                    print(region_data_hist["daily_deaths"].values[-1])
                    
                    print("----")'''
            
            ##################
            
            growth_rates.append(growth_rate_c)
            growth_rates_deaths.append(growth_rate_d)
                
            total_cases.append(float(region_data["confirmed"].values[-1]))
            
            #if is_pred == "Actual":
            
            
            daily_cases.append(region_data["confirmed"].diff().values[-WINDOW_RISK:].mean())
            daily_deaths.append(region_data["deaths"].diff().values[-WINDOW_RISK:].mean())
            #if is_pred == 'Predicted':
            #daily_cases.append(region_data["confirmed"].diff().values[-1])
            #daily_deaths.append(region_data["deaths"].diff().values[-1])

        
        #old method
        #C = error_forget_factor**power_error_factor # error percentage adjustment
        #power_error_factor = power_error_factor + 1

        growth_rates = np.array(growth_rates)*np.array(errors_growth_rates)# np.array(growth_rates) + C*np.array(errors_growth_rates) # old method
        growth_rates_deaths = np.array(growth_rates_deaths)*np.array(errors_growth_rates_deaths)#np.array(growth_rates_deaths) + C*np.array(errors_growth_rates_deaths) # old method
        growth_rates[growth_rates<-0.5] = -0.5 # clip the negative value
        growth_rates_deaths[growth_rates_deaths<-0.5] = -0.5 # clip the negative value
        regions["is_TD"] = IS_TD
        regions["growth_rate"] = growth_rates
        regions["growth_rate_deaths"] = growth_rates_deaths
        Re = (1 + growth_rates*5 )*np.array(errors_Re) #(1 + growth_rates*5 ) + C*np.array(errors_Re) # old method
        Re[Re<0] = 0
        regions["Re"] = Re
        regions["total_cases"] = total_cases*np.array(errors_total_cases) #total_cases + C*np.array(errors_total_cases) # old method
        regions["total_cases_per_M"] = 1000000*regions["total_cases"]/regions["population"]
        daily_cases = np.array(daily_cases)*np.array(errors_daily_cases) #np.array(daily_cases) + C*np.array(errors_daily_cases) # old method
        daily_cases[daily_cases<0]=0
        daily_deaths = np.array(daily_deaths)*np.array(errors_daily_deaths)#np.array(daily_deaths) + C*np.array(errors_daily_deaths) # old method
        daily_deaths[daily_deaths<0]=0
        regions["daily_cases"] = daily_cases
        regions["daily_cases_per_M"] = 1000000*regions["daily_cases"]/regions["population"]
        regions["daily_deaths"] = daily_deaths
        regions["daily_deaths_per_M"] = 1000000*regions["daily_deaths"]/regions["population"]
        regions["is_pred"] = is_pred
        REGIONS = pd.concat((REGIONS,regions),axis=0)

        '''print(date_to_calc)
        print(region)
        print(error_daily_cases)
        print(region_data["confirmed"].diff().values[-WINDOW_RISK:].mean() )
        print(region_data_hist["daily_cases"].values[-1])
        print(daily_deaths)
        print("----")'''
        
        if day_risk % 7 == 0:
            print(int(100*day_risk/380.0),"% done")

        day_risk = day_risk + 1


    REGIONS["date_of_calc"] = MAX_DATE_CASES.strftime("%Y-%m-%d")


    # fill the NaNs for growth rates and Re with interpolation
    REGIONS_NEW = pd.DataFrame([],columns=['Country/Region', 'country_of_state', 'population', 'date', 'is_TD',
           'growth_rate', 'growth_rate_deaths', 'Re', 'total_cases',
           'total_cases_per_M', 'daily_cases', 'daily_cases_per_M', 'daily_deaths',
           'daily_deaths_per_M'],dtype=object)
    for row in regions[["Country/Region","country_of_state","population"]].values:
        region,country_of_state,pop = row
        #print(region,country_of_state)
        df_new = REGIONS[REGIONS["Country/Region"]==region].query("country_of_state == '{}'".format(country_of_state)).reset_index()
        df_new["growth_rate"] = df_new["growth_rate"].interpolate().values
        df_new["growth_rate_deaths"] = df_new["growth_rate_deaths"].interpolate().values
        df_new["Re"] = df_new["Re"].interpolate().values
        del df_new["index"]
        REGIONS_NEW = pd.concat((REGIONS_NEW,df_new),axis=0)

    create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
    copy_to_sql(df=REGIONS_NEW,table_name="regional_intensity_predictions",schema_name="rto",if_exists="append",primary_index="Country/Region")
    remove_context()
    logger.info("rto.regional_intensity_predictions done. Risk calculation for future 360 days.")
    print("rto.regional_intensity_predictions updated.")
else:
    print("rto.regional_intensity_predictions already updated.")

###################################################################################################################################


################################################# update the phase timelines table #############################################
# to check if the table is already updated or not.
connection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
cur = connection.cursor()
sql_command = """select max(date_of_calc) from rto.intensity_fixed_params where "Country/Region" = 'Wake';"""
cur.execute(sql_command)
res = cur.fetchall()
cur.close()
max_date_phase_tab = res[0][0]
#

if max_date_phase_tab != MAX_DATE_CASES.strftime("%Y-%m-%d"):

    connection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")

    cur = connection.cursor()

    sql_command = """
    insert into rto.intensity_fixed_params
    select * from
    (select
    "date",
    "Country/Region",
    country_of_state,
    date_of_calc,
    daily_cases_per_M,
    daily_deaths_per_M,
    growth_rate,
    growth_rate_deaths,
    Re,

    -LN((cast(2.0 as float)/(0.99+1.0))-1)/(20.0) as alpha_cases_sql
    ,(2.0 / (1 + EXP(-alpha_cases_sql*(daily_cases_per_M/10)))) - 1.0 as risk_factor_cases_sql

    ,-LN((cast(2.0 as float)/(0.99+1.0))-1)/(1.0) as alpha_deaths_sql
    ,(2.0 / (1 + EXP(-alpha_deaths_sql*(daily_deaths_per_M/10)))) - 1.0 as risk_factor_deaths_sql

    ,-LN((cast(1.0 as float)/(0.99))-1)/(0.2) as alpha_growth_rate_sql
    ,(1.0 / (1 + EXP(-alpha_growth_rate_sql*(growth_rate)))) as risk_factor_growth_rate_sql

    ,-LN((cast(1.0 as float)/(0.99))-1)/(0.2) as alpha_growth_rate_deaths_sql
    ,(1.0 / (1 + EXP(-alpha_growth_rate_deaths_sql*(growth_rate_deaths)))) as risk_factor_growth_rate_deaths_sql

    ,-LN((cast(1.0 as float)/(0.99))-1)/(1.4-1.0) as alpha_Re_sql
    ,(1.0 / (1 + EXP(-alpha_Re_sql*(Re-1.0)))) as risk_factor_Re_sql

    ,-LN((cast(2.0 as float)/(0.99+1.0))-1)/(0.75) as alpha_vacc_sql
    ,1  - ( (2.0 / (1 + EXP(-alpha_vacc_sql*pred_vacc_perc))) - 1.0) as risk_factor_vacc_sql


    ,(1*risk_factor_cases_sql+
    0*risk_factor_cases_sql*risk_factor_growth_rate_sql + 
    1*risk_factor_deaths_sql+
    0*risk_factor_deaths_sql*risk_factor_growth_rate_deaths_sql + 
    0*risk_factor_cases_sql*risk_factor_Re_sql +
    3*risk_factor_vacc_sql)/
    (1+0+1+0+0+3) as risk

    from rto.regional_intensity_pred_with_vacc_view

    ) tmp1
    inner join 

    (select site_id
    ,country_region
    ,state_province
    ,county_district
    ,city
    ,"Country/Region" as "Country/Region2"  
    ,country_of_state as country_of_state2
    ,population as population
    ,date_of_calc as date_of_calc2
    ,is_TD as is_TD2
    ,daily_cases_per_M as daily_cases_per_M_current
    ,daily_deaths_per_M as daily_deaths_per_M_current
    ,Re as Re_current
    ,growth_rate as growth_rate_current
    ,growth_rate_deaths as growth_rate_deaths_current
    ,granularity
    from rto.regional_intensity_profiling_future_view) tmp5
    on tmp5."Country/Region2" = tmp1."Country/Region" and tmp5.country_of_state2 = tmp1.country_of_state and tmp5.date_of_calc2 = date_of_calc;
    """

    cur.execute(sql_command)
    res = cur.fetchall()
    logger.info("rto.intensity_fixed_params done. Updated the phase timelines table.")
    print("rto.intensity_fixed_params updated.")
else:
    print("rto.intensity_fixed_params already updated.")


################################################### Vaccincation Data ###############################################

connection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
cur = connection.cursor()

sql_command = """
select tmp.city
    ,tmp.country_region
    ,tmp.country_of_state
    ,date_vacc,tmp.total_vaccinations_new
    ,people_vaccinated
    ,pop_country
    ,people_vacc_percent
    ,max_date_calc
from 
(
select city
    ,country_region
    ,cast('' as varchar(30)) as country_of_state
    ,date_vacc
    --,total_vaccinations
    ,total_vaccinations_new
    ,case when col_to_use = 'daily_vaccinations' then 0.3*total_vaccinations_new else people_vaccinated2 end as people_vaccinated
    ,pop_country
    ,case when col_to_use = 'daily_vaccinations' then 0.3*total_vaccinations_new/pop_country else people_vacc_percent2 end as people_vacc_percent
    from 
(
    select c.city, b.country_region,"date" as date_vacc,total_vaccinations,--col_to_use,
    --case when people_vaccinated is null then 0.56*total_vaccinations else people_vaccinated end people_vaccinated2,
    total_vaccinations_new,
    people_fully_vaccinated as people_vaccinated2,
    pop_country,
    people_fully_vaccinated/pop_country as people_vacc_percent2,
    col_to_use
    --concat(cast(cast(round(100*people_vacc_percent,0) as int) as varchar(3)),'%') as people_vacc_percent_str
    from (
    select a.*,case when country_region = 'Czechia' then 'Czech Republic' else country_region end as country_region_corrected,
    row_number() over(partition by country_region order by "date" desc) as row_num 
    from (select v1.*,sum(daily_vaccinations) over (partition by country_region order by "date" rows between unbounded preceding and current row) as total_vaccinations_new from rto.vaccinations_global v1 where "date" <= (select max("date") from rto.regional_intensity_profiling)) a
    ) b 
    
    inner join rto.td_sites c on b.country_region_corrected = c.country_region
    inner join rto.country_population d on b.country_region_corrected = d.country
    inner join rto.vacc_col_to_use e on b.country_region = e.country_region
    where e.country_region <> 'India'
) country_data

union all

select city
    ,country_region
    ,cast('India_district' as varchar(30)) as country_of_state
    ,cast("date" as date)
    --,total_vaccinations
    ,first_dose_admin as total_vaccinations_new
    ,second_dose_admin as people_vaccinated
    ,population
    ,cast(second_dose_admin as float)/population as people_vacc_percent
from rto.india_vacc_regional

union all

-- US vaccinations
select city,location_corrected,'US' as country_of_state, "date",total_vaccinations,people_fully_vaccinated,population,
people_fully_vaccinated/population as people_vacc_percent
from 
(
select a.*,case when location = 'New York State' then 'New York' else location end as location_corrected,b.* 
from (select * from rto.vaccinations_us where "date" <= (select max("date") from rto.regional_intensity_profiling) ) a 
inner join rto.td_sites b on location_corrected = b.state_province
) bb

inner join (select state,sum(population) as population from rto.population_us group by 1) c
on bb.location_corrected = c.state
) tmp

left join

(select city,country_region,country_of_state,max(date_of_calc) as max_date_calc from rto.vacc_predictions group by 1,2,3) v_check
on tmp.city = v_check.city and tmp.country_region = v_check.country_region and tmp.country_of_state = v_check.country_of_state

order by date_vacc;select tmp.city
    ,tmp.country_region
    ,tmp.country_of_state
    ,date_vacc,tmp.total_vaccinations_new
    ,people_vaccinated
    ,pop_country
    ,people_vacc_percent
    ,max_date_calc
from 
(
select city
    ,country_region
    ,cast('' as varchar(30)) as country_of_state
    ,date_vacc
    --,total_vaccinations
    ,total_vaccinations_new
    ,case when col_to_use = 'daily_vaccinations' then 0.3*total_vaccinations_new else people_vaccinated2 end as people_vaccinated
    ,pop_country
    ,case when col_to_use = 'daily_vaccinations' then 0.3*total_vaccinations_new/pop_country else people_vacc_percent2 end as people_vacc_percent
    from 
(
    select c.city, b.country_region,"date" as date_vacc,total_vaccinations,--col_to_use,
    --case when people_vaccinated is null then 0.56*total_vaccinations else people_vaccinated end people_vaccinated2,
    total_vaccinations_new,
    people_fully_vaccinated as people_vaccinated2,
    pop_country,
    people_fully_vaccinated/pop_country as people_vacc_percent2,
    col_to_use
    --concat(cast(cast(round(100*people_vacc_percent,0) as int) as varchar(3)),'%') as people_vacc_percent_str
    from (
    select a.*,case when country_region = 'Czechia' then 'Czech Republic' else country_region end as country_region_corrected,
    row_number() over(partition by country_region order by "date" desc) as row_num 
    from (select v1.*,sum(daily_vaccinations) over (partition by country_region order by "date" rows between unbounded preceding and current row) as total_vaccinations_new from rto.vaccinations_global v1 where "date" <= (select max("date") from rto.regional_intensity_profiling)) a
    ) b 
    
    inner join rto.td_sites c on b.country_region_corrected = c.country_region
    inner join rto.country_population d on b.country_region_corrected = d.country
    inner join rto.vacc_col_to_use e on b.country_region = e.country_region
    where e.country_region <> 'India'
) country_data

union all

select city
    ,country_region
    ,cast('India_district' as varchar(30)) as country_of_state
    ,cast("date" as date)
    --,total_vaccinations
    ,first_dose_admin as total_vaccinations_new
    ,second_dose_admin as people_vaccinated
    ,population
    ,cast(second_dose_admin as float)/population as people_vacc_percent
from rto.india_vacc_regional

union all

-- US vaccinations
select city,location_corrected,'US' as country_of_state, "date",total_vaccinations,people_fully_vaccinated,population,
people_fully_vaccinated/population as people_vacc_percent
from 
(
select a.*,case when location = 'New York State' then 'New York' else location end as location_corrected,b.* 
from (select * from rto.vaccinations_us where "date" <= (select max("date") from rto.regional_intensity_profiling) ) a 
inner join rto.td_sites b on location_corrected = b.state_province
) bb

inner join (select state,sum(population) as population from rto.population_us group by 1) c
on bb.location_corrected = c.state
) tmp

left join

(select city,country_region,country_of_state,max(date_of_calc) as max_date_calc from rto.vacc_predictions group by 1,2,3) v_check
on tmp.city = v_check.city and tmp.country_region = v_check.country_region and tmp.country_of_state = v_check.country_of_state

order by date_vacc;
"""

cur.execute(sql_command)
res = cur.fetchall()
cur.close()

df_vacc = pd.DataFrame(res,columns=np.array(cur.description)[:,0])
df_vacc["date_vacc"] = pd.to_datetime(df_vacc["date_vacc"])
#df_vacc["daily_vacc_per_pop"] = df_vacc["daily_vaccinations"]/df_vacc["population"]
df_vacc



##### Vacc Predictions

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression

def normalize_vacc_sig(x,Threshold = 0.5):
    lamda = 2.0
    gamma = 1.0
    alpha = -np.log((lamda/(Threshold+gamma))-1)/(Threshold)
    return lamda / (1 + np.exp(-alpha*(x))) - gamma

clf = LinearRegression()
window_for_averaging = 14 # used to calc error in pred and actual on the date of calc
error_forget_factor = 0.99
power_error_factor = 0
cnt_future_days = 366 # +1 for current date

df_vacc["date_vacc"] = pd.to_datetime(df_vacc["date_vacc"])

df_vacc_pred = pd.DataFrame([],columns=["city","country_region","country_of_state","date_of_calc","date_pred","pred_vacc_perc"],dtype=object)
for city,country_region,country_of_state,max_date_calc in df_vacc[["city","country_region","country_of_state","max_date_calc"]].drop_duplicates().sort_values("city").values:
    # 
    
    df_vacc.loc[df_vacc.query("city == '{}'".format(city)).index,"people_vacc_percent"] = df_vacc.query("city == '{}'".format(city))["people_vacc_percent"].fillna(method='ffill').values#.plot()
    dates_vacc = df_vacc.query("city == '{}'".format(city))["date_vacc"]
    days = dates_vacc.sub(datetime.datetime.strptime("2020-01-01","%Y-%m-%d")).dt.days
    date_of_calc = dates_vacc.iloc[-1].strftime("%Y-%m-%d")
    print(date_of_calc)
    train_data = df_vacc.query("city == '{}'".format(city))["people_vacc_percent"].values#.cumsum().values
    
    #print(date_of_calc)
    X = days.values[-30:]
    X = X.reshape(len(X),1)
    y = train_data[-30:]
    clf = LinearRegression()
    clf.fit(X.reshape(len(X),1),y.reshape(len(y),1))
    future_data = np.arange(days.values[-window_for_averaging-1],days.values[-1]+cnt_future_days)
    future_data = future_data.reshape(len(future_data),1)
    '''plt.figure()
    plt.title(city)
    plt.plot(days,train_data)
    plt.plot(future_data,clf.predict(future_data))'''
    
    #
    
    #print(mean_squared_error(y,clf.predict(X)), r2_score(y,clf.predict(X)))
    
    thresholds = np.arange(0.1,0.9,0.01)
    best_t = -1
    best_err = np.inf
    for t in thresholds:
        err_metric = mean_squared_error(y,normalize_vacc_sig(clf.predict(X),t))
        if err_metric < best_err:
            best_err = err_metric
            best_t = t
    #print(best_t)
    #print(np.sqrt(mean_squared_error(y,normalize_vacc_sig(clf.predict(X),best_t))), r2_score(y,normalize_vacc_sig(clf.predict(X),best_t)))
    #plt.plot(days[-30:],normalize_vacc_sig(clf.predict(X),best_t))
    future_vacc_predictions = normalize_vacc_sig(clf.predict(future_data),best_t)
    
    #print(X[-1])
    #print(future_data[window_for_averaging])
    
    error_vacc = y[-1] - future_vacc_predictions[window_for_averaging]
    #print(future_vacc_predictions[window_for_averaging],y[-1],error_vacc)
    
    
    ef = [error_forget_factor**i for i in range(len(future_vacc_predictions)-(window_for_averaging))]
    ef = list(np.zeros(window_for_averaging) ) + ef
    #print(future_vacc_predictions.shape,np.array(ef).reshape(len(ef),1).shape)
    future_vacc_predictions = np.array(future_vacc_predictions) + np.array(ef).reshape(len(ef),1)*error_vacc
    
    
    '''plt.plot(future_data,future_vacc_predictions)
    plt.show()
    plt.grid()
    plt.close()'''
    
    
    #print(future_vacc_predictions.shape,future_data.shape)
    start_pred_date = (dates_vacc.dt.date.values[-window_for_averaging-1]).strftime("%Y-%m-%d")
    future_dates = pd.date_range(start=start_pred_date,freq='24H',periods=len(future_data))
    #print(future_vacc_predictions.shape,future_dates.shape,start_pred_date,future_data.shape)
    
    '''plt.figure()
    plt.title(city)
    plt.plot(days,train_data)
    plt.plot(future_data,future_vacc_predictions)
    plt.show()
    plt.close()'''
    
    
    '''print(start_pred_date,dates_vacc.iloc[-1])
    print(future_data[window_for_averaging:].shape,future_data[window_for_averaging:].shape)
    print(future_dates)
    print(pd.datetime.datetime.strptime("2020-01-01","%Y-%m-%d")+datetime.timedelta(days=529))
    print(start_pred_date)'''
    
    
    df_v_p_tmp = pd.DataFrame([],columns=["city","country_region","country_of_state","date_of_calc","date_pred","pred_vacc_perc"],dtype=object)
    
    
    df_v_p_tmp["pred_vacc_perc"] = future_vacc_predictions.flatten()
    df_v_p_tmp["date_of_calc"] = date_of_calc
    df_v_p_tmp["date_pred"] = future_dates.date
    df_v_p_tmp["city"] = city
    df_v_p_tmp["country_region"] = country_region
    
    df_v_p_tmp["country_of_state"] = country_of_state
    #print(date_of_calc,max_date_calc)
    if max_date_calc == None or date_of_calc > max_date_calc:
        df_vacc_pred = pd.concat((df_vacc_pred,df_v_p_tmp),axis=0)
    
    print("vacc",city)
    #break
try:
    
    create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
    copy_to_sql(df=df_vacc_pred,table_name="vacc_predictions",schema_name="rto",if_exists="append")
    remove_context()
    command_status = "rto.vacc_predictions updated. Predictions (Vaccinations) done."
    #logger.info("rto.vacc_predictioins updated. Predictions (Vaccinations) done.")
except Exception as e:
    command_status = "error updating rto.vacc_predictions."
    remove_context()
print(command_status)
logger.info(command_status)

# save the current vacc data to plot in the trend analysis tab. 
df_vacc["date_vacc"] = df_vacc["date_vacc"].dt.date.astype(str)
create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
copy_to_sql(df=df_vacc,table_name="vacc_current",schema_name="rto",if_exists="replace")
remove_context()








print("Script executed Successfully!")


logger.info("Script executed Successfully!")

