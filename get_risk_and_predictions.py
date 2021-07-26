'''
Version: optimizing on daily difference with death correction and SG_Filter, optimizing with vacc data as well
This code has been tested on Python3.6 and can take more that hour to execute.
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

delete from rto.regional_intensity_profiling where "date">'2021-06-29';
delete from rto.LG_predicted_cases where date_of_calc >'2021-06-29';
delete from rto.LG_predicted_cases_with_conf where date_of_calc >'2021-06-29';
delete from rto.LG_predicted_deaths where date_of_calc >'2021-06-29';
delete from rto.regional_intensity_profiling_future where date_of_calc >'2021-06-29';
delete from rto.regional_intensity_predictions where date_of_calc >'2021-06-29';
delete from rto.intensity_fixed_params where date_of_calc > '2021-06-29';
'''
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

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
fh = logging.FileHandler('rto_script.log')
fh.setFormatter(formatter)
logger.addHandler(fh)
#logger.info("starting")
#exit()

def func(t, A, lamda): # y = A*exp(lambda*t)
    y = A*np.exp(lamda*t)
    return y
ROLLING_MEAN_FOR_GROWTH_CALC = 0
def cost_function(params):
    y = func(np.arange(0,WINDOW,1),params[0],params[1])
    #print("y",y,D["JHU_ConfirmedCases.data"].diff().values[-WINDOW:])
    assert ROLLING_MEAN_FOR_GROWTH_CALC ==0 or ROLLING_MEAN_FOR_GROWTH_CALC ==1

     
    Ddiff = D.diff().fillna(value=D.diff().mean()) # in case the first value is NAN
    if ROLLING_MEAN_FOR_GROWTH_CALC ==0:
        return np.sum((y - Ddiff.values[-WINDOW:])**2)
    elif ROLLING_MEAN_FOR_GROWTH_CALC ==1:
        return np.sum((y - Ddiff.rolling(window=14).mean().values[-WINDOW:])**2)
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





################################################# Fetch data from vantage #########################################################

# fetch data from vantage
print("Fetching data from Vantage")

import teradatasql
connection = teradatasql.connect(host="tdprd.td.teradata.com", user="RTO_SVC_ACCT", password="svcOct2020#1008")
cur = connection.cursor()

cur.execute("""
select main_data.*,vacc_data.pred_vacc_perc as vacc_perc from 
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
) main_data

left join (
select distinct "Country/Region",country_of_state, pred_vacc_perc from rto.regional_intensity_pred_with_vacc_view 
where date_of_calc = (select max(date_of_calc) from rto.regional_intensity_pred_with_vacc_view) and date_of_calc = "date"
) vacc_data
on main_data."Country/Region" = vacc_data."Country/Region" and main_data.country_of_state = vacc_data.country_of_state

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
cur.execute("""select cast(max("date") as date)+1 from rto.regional_intensity_profiling""")
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
    )as tmp;""")
    max_date_cases_tab = cur.fetchall()[0][0]
if type(max_date_cases_tab) == datetime.date:
    max_date_cases_tab = max_date_cases_tab.strftime("%Y-%m-%d")

cur.close()

#max_date_region_int_tab = '2020-10-18' # CHANGE THIS OR COMMENT IT IF NEEDED TO RECALCULATE
#max_date_cases_tab = '2021-02-27' # CHANGE THIS OR COMMENT IT IF NEEDED TO RECALCULATE

print(max_date_region_int_tab,max_date_cases_tab)
if max_date_cases_tab>=max_date_region_int_tab:
    dates_to_calc= pd.date_range(start=max_date_region_int_tab, end = max_date_cases_tab)
    print(dates_to_calc)
    
    REGIONS = pd.DataFrame([],columns=['Country/Region', 'country_of_state', 'population', 'date', 'is_TD',
           'growth_rate', 'growth_rate_deaths', 'Re', 'total_cases',
           'total_cases_per_M', 'daily_cases', 'daily_cases_per_M', 'daily_deaths',
           'daily_deaths_per_M'])
    for date_to_calc in dates_to_calc:
        total_cases = []
        daily_cases = []
        daily_deaths = []
        growth_rates = []
        growth_rates_deaths = []
        WINDOW = 14
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
            print(region,country_of_state,pop)
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
            print(region,growth_rate)


            D = region_data["deaths"]#D = COUNTRY_DATA[country]["JHU_ConfirmedDeaths.data"]
            initial_guess = [A,lamda]
            result = optimize.fmin(cost_function,initial_guess,maxfun=1000,maxiter=1000)
            A_,growth_rate = result 
            growth_rates_deaths.append(growth_rate)

            total_cases.append(float(region_data["confirmed"].values[-1]))
            daily_cases.append(region_data["confirmed"].diff().values[-WINDOW:].mean())
            daily_deaths.append(region_data["deaths"].diff().values[-WINDOW:].mean())
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
    print("Script already executed till the latest date")
    sys.exit()
#REGIONS_CURRENT = REGIONS

##################################################################################################################################
logger.info("rto.regional_intensity_profiling' done. Risk Calculated till the latest date.")

###################################################### for prediction modeling #####################################################

# for prediction modeling

from scipy.optimize import Bounds
from scipy import stats

forget_factor = 0.9
WINDOW = 30
def get_predictions_sigmoid(x,alpha,lamda = 1,beta = 0):
    cases = (lamda / (1 + np.exp(-alpha*(x-beta))))
    return cases
def cost_predictions(params):
    y = get_predictions_sigmoid(np.arange(0,len(actual),1),params[0],params[1],params[2])
    
    # in case fitting on diff is required
    y = np.diff(y)
    actual2= np.diff(actual)
    actual2[actual2<0]=0
    #actual = np.diff(actual)
    f = [forget_factor**i for i in range(len(actual2))][::-1]
    
    return np.sum(f[-WINDOW:]*(y[-WINDOW:] - actual2[-WINDOW:])**2)

def cost_actual(params):
    y = get_predictions_sigmoid(np.arange(0,len(actual)+365,1),params[0],params[1],params[2])
    is_growing = False
    # in case fitting on diff is required
    y = np.diff(y)
    actual2= np.diff(actual)
    #actual = np.diff(actual)
    win_actual = 2*window_for_averaging
    start_pred_val = y[-win_actual] 
    max_pred_val = max(y)
    if start_pred_val < max_pred_val:
        is_growing = True
    
    
    last_actual_val = np.max(np.diff(actual_all)[-win_actual:])
    #print(start_pred_val,max_pred_val,last_actual_val,len(y))
    #if is_growing == True and last_actual_val > max_pred_val and last_actual_val > 100:
    #    return 100000000000000
        
    return np.sum((y[len(actual)-win_actual:len(actual)] - actual2[-win_actual:])**2)

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


###################################################################################################################################


############################################################# Case predictions: ####################################################

from matplotlib.backends.backend_pdf import PdfPages  
pp = PdfPages('predictions_{}.pdf'.format(str(max_date_cases_tab)))

# Case predictions:
print("Calculating Predictions")

window_for_averaging = 15#14

create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")

max_date_previous=DataFrame.from_query("select max(date_of_calc) as max_date_previous from rto.LG_predicted_cases").to_pandas()
max_date_previous=max_date_previous["max_date_previous"].values[0]

if len(sys.argv) == 2:
    MAX_DATE_CASES = datetime.datetime.strptime(sys.argv[1],"%Y-%m-%d").date()
else:
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
    )as tmp;""").to_pandas().iloc[0,0]


dates_to_pred= pd.date_range(start=(max_date_previous+datetime.timedelta(days=1)).strftime("%Y-%m-%d"), 
                             end = MAX_DATE_CASES.strftime("%Y-%m-%d"))#str(np.datetime_as_string(MAX_DATE_CASES,unit='D'))
dates_to_pred

END_DATES_COUNTRIES = pd.DataFrame([],columns=["Country/Region","country_of_state","date_of_calc","pred_end_date","pred_days_remaining_in_epidemic"])
PRED_CASES_COUNTRIES = pd.DataFrame([],columns=["Country/Region","country_of_state","date_of_calc","date","pred_confirmed_cases"])

regions = df_cases[["Country/Region","country_of_state","population","vacc_perc"]].drop_duplicates(subset=["Country/Region","country_of_state"]).sort_values(by="Country/Region")

#create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
#dates_to_pred= pd.date_range(start='2020-10-17', end = '2020-12-02')
print(dates_to_pred)
for date_to_pred in dates_to_pred:
    max_date_cases = date_to_pred.strftime("%Y-%m-%d")
    #max_date_cases = df_cases["date"].values[-1]
    #print(max_date_cases)
    end_dates_countries = []
    predicted_cases_countries = pd.DataFrame([],columns=["Country/Region","country_of_state","date_of_calc","date","pred_confirmed_cases"])
    WINDOWS = [60]#np.arange(30,60)
    FORGET_FACTORS=[0.99,0.9,0.85]#[0.9,0.95,0.99]#0.999,
    MA_WINDOWS = [3,7,15]#[2,7,14]
        
    #for country in regions.query("country_of_state == 'US'")["Country/Region"].values:#["United States"]:#countries
    for country,country_of_state,population,vacc_perc in regions.values:
        #sprint(date_to_pred,country,"_",vacc_perc)
        actual_all = df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["confirmed"].values
        actual = actual_all#[0:i]
        if vacc_perc > 1:
            vacc_perc = 0.1
            
        suscepible_pop = population - (np.max(actual) + vacc_perc*population)
        bounds = Bounds([0, np.max(actual),0], [0.1, suscepible_pop,1000])#np.max(actual)/max_infected
            
        PREDICTIONS = []
        best_score = np.inf
        alpha_best,lamda_best,beta_best=0.02,0,0
        win_best,fg_best =0,0
        for ma_win in MA_WINDOWS:
            for fg in FORGET_FACTORS:
                forget_factor = fg

                for win in WINDOWS:
                    WINDOW = win
                    #actual = pd.Series(actual_all).rolling(window=ma_win).mean()#[0:i]
                    actual = savgol_filter(actual_all, ma_win, 2)
                    res = optimize.minimize(fun=cost_predictions,x0=[0.05,np.max(actual),200],bounds=bounds,method="L-BFGS-B")#method="Nelder-Mead")#,method='Nelder-Mead')
                    alpha,lamda,beta = res.x
                    current_score = cost_actual([alpha,lamda,beta])
                    #print(ma_win,win,fg,alpha,lamda,beta,current_score)
                    if current_score < best_score:# and (alpha_best > 0.01 or alpha > 0.01):
                        best_score = current_score
                        alpha_best,lamda_best,beta_best = alpha,lamda,beta
                        win_best,fg_best = win,fg
        
        print(date_to_pred,country,"best",win_best,fg_best,alpha_best,lamda_best,beta_best,"best_score=",best_score)
        end_date = get_end_date(get_predictions_sigmoid(np.arange(0,len(dates),1),alpha_best,lamda_best,beta_best))
        end_dates_countries.append([country,"",max_date_cases,end_date[0],end_date[1]])
        predictions = get_predictions_sigmoid(np.arange(0,len(actual)+365,1)[len(actual)-window_for_averaging:],alpha_best,lamda_best,beta_best)
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

        p = predictions
        a = df_cases[df_cases["Country/Region"] ==country].query("country_of_state == '{}'".format(country_of_state))
        fig = plt.figure(figsize=[10,5])
        param_str = "w={},f={},a={:.4f},l={:.4f},b={:.4f}".format(win_best,fg_best,alpha_best,lamda_best,beta_best)
        plt.title("Cases {},{},{}".format(country,date_to_pred.strftime("%Y-%m-%d"),param_str))
        plt.plot(pd.to_datetime(p["date"]),p["pred_confirmed_cases"].diff())
        plt.plot(a["date"],a["confirmed"].diff())
        pp.savefig(fig)
        #plt.show()
        plt.close()

    end_dates_countries = pd.DataFrame(end_dates_countries,columns=["Country/Region","country_of_state","date_of_calc","pred_end_date","pred_days_remaining_in_epidemic"])
    
    PRED_CASES_COUNTRIES = pd.concat((PRED_CASES_COUNTRIES,predicted_cases_countries),axis=0)
    END_DATES_COUNTRIES = pd.concat((END_DATES_COUNTRIES,end_dates_countries),axis=0)
    
    
    #copy_to_sql(df=end_date_all,table_name="LG_predicted_end_dates",schema_name="rto",if_exists="append",primary_index="Country/Region")
    
    copy_to_sql(df=predicted_cases_countries,table_name="LG_predicted_cases",schema_name="rto",if_exists="append",primary_index="Country/Region")

remove_context()


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

create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")

PRED_DEATHS_COUNTRIES = pd.DataFrame([],columns=["Country/Region","country_of_state","date_of_calc","date","pred_deaths"])

regions = df_cases[["Country/Region","country_of_state","population"]].drop_duplicates(subset=["Country/Region","country_of_state"]).sort_values(by="Country/Region")

#create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")

#dates_to_pred= pd.date_range(start='2020-10-17', end = '2020-11-29')
for date_to_pred in dates_to_pred:
    max_date_cases = date_to_pred.strftime("%Y-%m-%d")
    #max_date_cases = df_cases["date"].values[-1]
    print(max_date_cases)
    end_dates_countries = []
    predicted_deaths_countries = pd.DataFrame([],columns=["Country/Region","country_of_state","date_of_calc","date","pred_deaths"])
    WINDOWS = [60]#np.arange(30,60)
    FORGET_FACTORS=[0.99,0.9,0.85]#[0.9,0.95,0.99]
    MA_WINDOWS = [3,7,15]#[2,7,14]
        
    #for country in regions.query("country_of_state == 'US'")["Country/Region"].values:#["United States"]:#countries
    for country,country_of_state,population in regions.values:
        #print(date_to_pred,country)
        actual_all = df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["deaths"].values
        #actual_all = np.nan_to_num(df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["deaths"].rolling(window=14).mean().values)
        actual = actual_all#[0:i]

        # for setting the max bound for deaths based on case predictions
        cases = df_cases[df_cases["Country/Region"]==country].query("country_of_state == '"+country_of_state+"' and date <= '"+max_date_cases+"'")["confirmed"].values
        mortality_rate = actual_all[-1]/cases[-1]
        pred_cases = PRED_CASES_COUNTRIES[PRED_CASES_COUNTRIES["Country/Region"]==country].query("country_of_state == '"+country_of_state+"'")["pred_confirmed_cases"].values
        bounds = Bounds([0, actual_all[-1],0], [0.1, mortality_rate*pred_cases[-1],1000])#np.max(actual)/max_infected
        
            
        PREDICTIONS_DEATHS = []
        best_score = np.inf
        alpha_best,lamda_best,beta_best=0.02,0,0
        win_best,fg_best =0,0
        for ma_win in MA_WINDOWS:
            for fg in FORGET_FACTORS:
                forget_factor = fg

                for win in WINDOWS:
                    WINDOW = win
                    #actual = pd.Series(actual_all).rolling(window=ma_win).mean()#[0:i]
                    actual = savgol_filter(actual_all, ma_win, 2)
                    res = optimize.minimize(fun=cost_predictions,x0=[0.05,actual_all[-1],200],bounds=bounds,method="L-BFGS-B")
                    alpha,lamda,beta = res.x
                    current_score = cost_actual([alpha,lamda,beta])
                    #print(ma_win,win,fg,alpha,lamda,beta,current_score)
                    if current_score < best_score:# and (alpha_best > 0.01 or alpha > 0.01):
                        best_score = current_score
                        alpha_best,lamda_best,beta_best = alpha,lamda,beta
                        win_best,fg_best = win,fg
                    
        print(date_to_pred,country,"best",win_best,fg_best,alpha_best,lamda_best,beta_best,"best_score=",best_score)       
        end_date = get_end_date(get_predictions_sigmoid(np.arange(0,len(dates),1),alpha_best,lamda_best,beta_best))
        end_dates_countries.append([country,"",max_date_cases,end_date[0],end_date[1]])
        predictions_deaths = get_predictions_sigmoid(np.arange(0,len(actual)+365,1)[len(actual)-window_for_averaging:],alpha_best,lamda_best,beta_best)
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

        p = predictions_deaths
        a = df_cases[df_cases["Country/Region"] ==country].query("country_of_state == '{}'".format(country_of_state))
        
        fig = plt.figure(figsize=[10,5])
        param_str = "w={},f={},a={:.4f},l={:.4f},b={:.4f}".format(win_best,fg_best,alpha_best,lamda_best,beta_best)
        plt.title("Deaths {},{},{}".format(country,date_to_pred.strftime("%Y-%m-%d"),param_str))
        plt.plot(pd.to_datetime(p["date"]),p["pred_deaths"].diff())
        plt.plot(a["date"],a["deaths"].diff())
        pp.savefig(fig)
        plt.close()
        #plt.show()

    end_dates_countries = pd.DataFrame(end_dates_countries,columns=["Country/Region","country_of_state","date_of_calc","pred_end_date","pred_days_remaining_in_epidemic"])
    
    PRED_DEATHS_COUNTRIES = pd.concat((PRED_DEATHS_COUNTRIES,predicted_deaths_countries),axis=0)
        
    
    copy_to_sql(df=predicted_deaths_countries,table_name="LG_predicted_deaths",schema_name="rto",if_exists="append",primary_index="Country/Region")

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
    (select * from rto.td_sites where country_region<>'US' and country_region <>'India') bb
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

) aaa

inner join
(
select bbb."Country/Region",bbb.country_of_state,bbb."date",bbb.pred_confirmed_cases,ccc.pred_deaths from 
(select * from rto.LG_predicted_cases where country_of_state in ('','US_county','India','India_district') and date_of_calc = (select max(date_of_calc) from rto.LG_predicted_cases)) bbb
inner join
(select * from rto.LG_predicted_deaths where country_of_state in ('','US_county','India','India_district') and date_of_calc = (select max(date_of_calc) from rto.LG_predicted_deaths)) ccc
on bbb."date" = ccc."date" and bbb."Country/Region" = ccc."Country/Region" and bbb.country_of_state = ccc.country_of_state
) ddd
on aaa."Country/Region" = ddd."Country/Region"
order by "date";
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

# calculate risk on predicted data
ROLLING_MEAN_FOR_GROWTH_CALC = 0
REGIONS = pd.DataFrame([],columns=['Country/Region', 'country_of_state', 'population', 'date', 'is_TD',
       'growth_rate', 'growth_rate_deaths', 'Re', 'total_cases',
       'total_cases_per_M', 'daily_cases', 'daily_cases_per_M', 'daily_deaths',
       'daily_deaths_per_M'])
for date_to_calc in dates_to_calc:
    total_cases = []
    daily_cases = []
    daily_deaths = []
    growth_rates = []
    growth_rates_deaths = []
    WINDOW = 14
    A = 1000
    lamda = 0.001

    regions = df_cases_pred[["Country/Region","country_of_state","population"]].drop_duplicates(subset=["Country/Region","country_of_state"]).sort_values(by="Country/Region")
    max_date_cases = df_cases_pred["date"].values[-1]
    regions["date"] = date_to_calc.date().strftime("%Y-%m-%d")#max_date_cases
    IS_TD = []
    print(date_to_calc)
    for row in regions[["Country/Region","country_of_state","population"]].values:

        region,country_of_state,pop = row
        print(region,country_of_state,pop)
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
        daily_cases.append(region_data["confirmed"].diff().values[-WINDOW:].mean())
        daily_deaths.append(region_data["deaths"].diff().values[-WINDOW:].mean())
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

create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
copy_to_sql(df=REGIONS_MERGED,table_name="regional_intensity_profiling_future",schema_name="rto",if_exists="append",primary_index="Country/Region")
remove_context()
REGIONS_MERGED

###################################################################################################################################
logger.info("rto.regional_intensity_profiling_future done. Risk prediction for specific future dates.")


################################################# risk predictions for all countries #############################################

print("risk predictions (365 days into the future)")
#df_cases_pred2 = df_cases[df_cases["Country/Region"].isin(df_cases_pred["Country/Region"].unique())]
#df_cases_pred2 = pd.concat((df_cases_pred2,df_cases_pred))
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
       'daily_deaths_per_M'])

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

for date_to_calc in dates_to_calc:
    total_cases = []
    daily_cases = []
    daily_deaths = []
    growth_rates = []
    growth_rates_deaths = []
    
    
    
    
    is_pred = []
    WINDOW = 14
    A = 1000
    lamda = 0.001

    regions = df_cases_pred2[["Country/Region","country_of_state","population"]].drop_duplicates(subset=["Country/Region","country_of_state"]).sort_values(by="Country/Region")
    max_date_cases = df_cases_pred2["date"].values[-1]
    regions["date"] = date_to_calc.date().strftime("%Y-%m-%d")#max_date_cases
    IS_TD = []
    print(date_to_calc)
    if date_to_calc.date() <= MAX_DATE_CASES:
        is_pred = "Actual" 
    else:
        is_pred = "Predicted"
        
    #regions = regions[(regions["Country/Region"] =='Japan') | (regions["Country/Region"] =='Pakistan') |(regions["Country/Region"]=='United Arab Emirates')]
    for row in regions[["Country/Region","country_of_state","population"]].values:

        region,country_of_state,pop = row
        print(region,country_of_state,pop)
        if region in TD_regions:
            IS_TD.append(1)
        else:
            IS_TD.append(0)
            
        

        #region_data = df_cases_pred2[df_cases_pred2["Country/Region"]== region].query("country_of_state == '"+str(country_of_state)+"'")
        region_data = df_cases_pred2[df_cases_pred2["Country/Region"]== region].query("country_of_state == '"+str(country_of_state)+"' and date <= '"+date_to_calc.date().strftime("%Y-%m-%d")+"'")
        
        
        
        
        
        D = region_data["confirmed"]
        
        
        initial_guess = [A,lamda]
        result = optimize.fmin(cost_function,initial_guess,maxfun=1000,maxiter=1000)
        A_,growth_rate_c = result 
        
        print(region,growth_rate_c)


        D = region_data["deaths"]#D = COUNTRY_DATA[country]["JHU_ConfirmedDeaths.data"]

        initial_guess = [A,lamda]
        result = optimize.fmin(cost_function,initial_guess,maxfun=1000,maxiter=1000)
        A_,growth_rate_d = result 
        
        ####### for error term to correct the predictions
        if date_to_calc.date() == MAX_DATE_CASES:
            ef = [error_forget_factor**i for i in range(len(region_data))]
            region_data_hist = REGIONS_CURRENT[REGIONS_CURRENT["Country/Region"]== region].query("country_of_state == '"+str(country_of_state)+"' and date == '"+date_to_calc.date().strftime("%Y-%m-%d")+"'")
            
            error_total_cases = float(region_data_hist["total_cases"].values[-1]) - float(region_data["confirmed"].values[-1])
            error_daily_cases = region_data_hist["daily_cases"].values[-1] - region_data["confirmed"].diff().values[-WINDOW:].mean() 
            error_daily_deaths = region_data_hist["daily_deaths"].values[-1] - region_data["deaths"].diff().values[-WINDOW:].mean()
            error_growth_rates = region_data_hist["growth_rate"].values[-1] - growth_rate_c 
            error_growth_rates_deaths = region_data_hist["growth_rate_deaths"].values[-1] - growth_rate_d
            error_Re = region_data_hist["Re"].values[-1] - ( 1 + growth_rate_c*5 )
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
        
        ##################
        
        growth_rates.append(growth_rate_c)
        growth_rates_deaths.append(growth_rate_d)
            
        total_cases.append(float(region_data["confirmed"].values[-1]))
        
        #if is_pred == "Actual":
        
        
        daily_cases.append(region_data["confirmed"].diff().values[-WINDOW:].mean())
        daily_deaths.append(region_data["deaths"].diff().values[-WINDOW:].mean())
        #if is_pred == 'Predicted':
        #daily_cases.append(region_data["confirmed"].diff().values[-1])
        #daily_deaths.append(region_data["deaths"].diff().values[-1])
    
    
    C = error_forget_factor**power_error_factor # error percentage adjustment
    power_error_factor = power_error_factor + 1

    growth_rates = np.array(growth_rates) + C*np.array(errors_growth_rates)
    growth_rates_deaths = np.array(growth_rates_deaths) + C*np.array(errors_growth_rates_deaths)
    growth_rates[growth_rates<-0.5] = -0.5 # clip the negative value
    growth_rates_deaths[growth_rates_deaths<-0.5] = -0.5 # clip the negative value
    regions["is_TD"] = IS_TD
    regions["growth_rate"] = growth_rates
    regions["growth_rate_deaths"] = growth_rates_deaths
    Re = (1 + growth_rates*5 ) + C*np.array(errors_Re)
    Re[Re<0] = 0
    regions["Re"] = Re
    regions["total_cases"] = total_cases + C*np.array(errors_total_cases)
    regions["total_cases_per_M"] = 1000000*regions["total_cases"]/regions["population"]
    daily_cases = np.array(daily_cases) + C*np.array(errors_daily_cases)
    daily_cases[daily_cases<0]=0
    daily_deaths = np.array(daily_deaths) + C*np.array(errors_daily_deaths)
    daily_deaths[daily_deaths<0]=0
    regions["daily_cases"] = daily_cases
    regions["daily_cases_per_M"] = 1000000*regions["daily_cases"]/regions["population"]
    regions["daily_deaths"] = daily_deaths
    regions["daily_deaths_per_M"] = 1000000*regions["daily_deaths"]/regions["population"]
    regions["is_pred"] = is_pred
    REGIONS = pd.concat((REGIONS,regions),axis=0)


REGIONS["date_of_calc"] = MAX_DATE_CASES.strftime("%Y-%m-%d")
create_context(host="tdprd.td.teradata.com",username="RTO_SVC_ACCT", password="svcOct2020#1008")
copy_to_sql(df=REGIONS,table_name="regional_intensity_predictions",schema_name="rto",if_exists="append",primary_index="Country/Region")
remove_context()


###################################################################################################################################
logger.info("rto.regional_intensity_predictions done. Risk calculation for future 360 days.")

################################################# update the phase timelines table #############################################

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


,(4*risk_factor_cases_sql+
1*risk_factor_cases_sql*risk_factor_growth_rate_sql + 
4*risk_factor_deaths_sql+
1*risk_factor_deaths_sql*risk_factor_growth_rate_deaths_sql + 
1*risk_factor_cases_sql*risk_factor_Re_sql +
4*risk_factor_vacc_sql)/
(4+1+4+1+1+4) as risk

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
    ,cast('' as varchar(10)) as country_of_state
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
) country_data

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

df_vacc_pred = pd.DataFrame([],columns=["city","country_region","country_of_state","date_of_calc","date_pred","pred_vacc_perc"])
for city,country_region,country_of_state,max_date_calc in df_vacc[["city","country_region","country_of_state","max_date_calc"]].drop_duplicates().sort_values("city").values:
    # 
    
    df_vacc.loc[df_vacc.query("city == '{}'".format(city)).index,"people_vacc_percent"] = df_vacc.query("city == '{}'".format(city))["people_vacc_percent"].fillna(method='ffill').values#.plot()
    dates_vacc = df_vacc.query("city == '{}'".format(city))["date_vacc"]
    days = dates_vacc.sub(pd.datetime.datetime.strptime("2020-01-01","%Y-%m-%d")).dt.days
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
    
    
    df_v_p_tmp = pd.DataFrame([],columns=["city","country_region","country_of_state","date_of_calc","date_pred","pred_vacc_perc"])
    
    
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

logger.info("rto.intensity_fixed_params done. Updated the phase timelines table.")
logger.info("Script executed Successfully!")

