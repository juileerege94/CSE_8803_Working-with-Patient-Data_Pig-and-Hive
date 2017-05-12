-- ***************************************************************************
-- TASK
-- Aggregate events into features of patient and generate training, testing data for mortality prediction.
-- Steps have been provided to guide you.
-- You can include as many intermediate steps as required to complete the calculations.
-- ***************************************************************************

-- ***************************************************************************
-- TESTS
-- To test, please change the LOAD path for events and mortality to ../../test/events.csv and ../../test/mortality.csv
-- 6 tests have been provided to test all the subparts in this exercise.
-- Manually compare the output of each test against the csv's in test/expected folder.
-- ***************************************************************************

-- register a python UDF for converting data into SVMLight format
REGISTER utils.py USING jython AS utils;

-- load events file
events = LOAD '../../data/events.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, eventdesc:chararray, timestamp:chararray, value:float);

-- select required columns from events
events = FOREACH events GENERATE patientid, eventid, ToDate(timestamp, 'yyyy-MM-dd') AS etimestamp, value;

-- load mortality file
mortality = LOAD '../../data/mortality.csv' USING PigStorage(',') as (patientid:int, timestamp:chararray, label:int);

mortality = FOREACH mortality GENERATE patientid, ToDate(timestamp, 'yyyy-MM-dd') AS mtimestamp, label;

--To display the relation, use the dump command e.g. DUMP mortality;

-- ***************************************************************************
-- Compute the index dates for dead and alive patients
-- ***************************************************************************
-- perform join of events and mortality by patientid;
eventswithmort = JOIN events BY patientid, mortality BY patientid; -- contains only dead patients

-- detect the events of dead patients and create it of the form (patientid, eventid, value, label, time_difference) 
-- where time_difference is the days between index date and each event timestamp
deadevents = FOREACH eventswithmort GENERATE events::patientid as patientid, events::eventid as eventid, events::value as value, mortality::label as label, DaysBetween(SubtractDuration(mtimestamp,'P30D'),etimestamp) as time_difference;

-- detect the events of alive patients and create it of the form (patientid, eventid, value, label, time_difference) 
-- where time_difference is the days between index date and each event timestamp
eventswithalive = JOIN events by patientid LEFT OUTER, mortality by patientid; -- all alive patients with null values for dead
aliveevents = FILTER eventswithalive by $4 is null; -- all alive patients having null in columns of m.patientid, mtimestamp and m.label
aliveevents = foreach aliveevents generate events::patientid as patientid, events::eventid as eventid, events::value as value, events::etimestamp as etimestamp, (mortality::label is null ? 0 : mortality::label) as label;
alive_group = GROUP aliveevents BY patientid;
temp_alive = foreach alive_group generate group as patientid, MAX(aliveevents.etimestamp) as index_date;
index_alive = JOIN aliveevents BY patientid, temp_alive BY patientid; -- index_data column attached to every patient
index_alive = FOREACH index_alive GENERATE aliveevents::patientid as patientid, aliveevents::eventid as eventid, aliveevents::value as value, aliveevents::label as label, DaysBetween(index_date,aliveevents::etimestamp) as time_difference;
aliveevents = index_alive;

--TEST-1
deadevents = ORDER deadevents BY patientid, eventid;
aliveevents = ORDER aliveevents BY patientid, eventid;
STORE aliveevents INTO 'aliveevents' USING PigStorage(',');
STORE deadevents INTO 'deadevents' USING PigStorage(',');

-- ***************************************************************************
-- Filter events within the observation window and remove events with missing values
-- ***************************************************************************
-- contains only events for all patients within the observation window of 2000 days and is of the form (patientid, eventid, value, label, time_difference)
filtered = UNION aliveevents, deadevents;
filtered = FILTER filtered BY time_difference<=2000;
filtered = FILTER filtered by $2 is not null;

--TEST-2
filteredgrpd = GROUP filtered BY 1;
filtered = FOREACH filteredgrpd GENERATE FLATTEN(filtered);
filtered = ORDER filtered BY patientid, eventid,time_difference;
STORE filtered INTO 'filtered' USING PigStorage(',');

-- ***************************************************************************
-- Aggregate events to create features
-- ***************************************************************************
-- for group of (patientid, eventid), count the number of  events occurred for the patient and create relation of the form (patientid, eventid, featurevalue)
filtered = foreach filtered generate filtered::patientid as patientid, filtered::eventid as eventid, filtered::value as value, filtered::label as label, filtered::time_difference as time_difference;
featureswithid = GROUP filtered BY (patientid, eventid);
featureswithid = FOREACH featureswithid GENERATE group.$0, group.$1 as eventid, COUNT(filtered.value) AS featurevalue;


--TEST-3
featureswithid = ORDER featureswithid BY patientid, eventid;
STORE featureswithid INTO 'features_aggregate' USING PigStorage(',');

-- ***************************************************************************
-- Generate feature mapping
-- ***************************************************************************
-- compute the set of distinct eventids obtained from previous step, sort them by eventid and then rank these features by eventid to create (idx, eventid). Rank should start from 0.
event_name = FOREACH featureswithid GENERATE eventid;
-- dump event_name;
-- describe event_name;
event_name = DISTINCT event_name;
event_name = ORDER event_name BY eventid;
all_features = RANK event_name;
all_features = foreach all_features generate (rank_event_name-1) as idx, eventid;

-- store the features as an output file
STORE all_features INTO 'features' using PigStorage(' ');


-- perform join of featureswithid and all_features by eventid and replace eventid with idx. It is of the form (patientid, idx, featurevalue)
features = JOIN featureswithid BY eventid, all_features BY eventid;
features = foreach features generate featureswithid::patientid as patientid, all_features::idx as idx, featureswithid::featurevalue as featurevalue;

--TEST-4
features = ORDER features BY patientid, idx;
STORE features INTO 'features_map' USING PigStorage(',');

-- ***************************************************************************
-- Normalize the values using min-max normalization
-- ***************************************************************************
-- group events by idx and compute the maximum feature value in each group. It is of the form (idx, maxvalues)
feature_group = GROUP features BY idx;
maxvalues = foreach feature_group generate group as idx, MAX(features.featurevalue) as maxvalues;

-- join features and maxvalues by idx
normalized = JOIN features by idx, maxvalues by idx;

-- compute the final set of normalized features of the form (patientid, idx, normalizedfeaturevalue)
features = foreach normalized generate features::patientid as patientid, features::idx as idx, (features::featurevalue / (double)maxvalues::maxvalues) as normalizedfeaturevalue;

--TEST-5
features = ORDER features BY patientid, idx;
STORE features INTO 'features_normalized' USING PigStorage(',');

-- ***************************************************************************
-- Generate features in svmlight format
-- features is of the form (patientid, idx, normalizedfeaturevalue) and is the output of the previous step
-- e.g.  1,1,1.0
--  	 1,3,0.8
--	 2,1,0.5
--       3,3,1.0
-- ***************************************************************************

grpd = GROUP features BY patientid;
grpd_order = ORDER grpd BY $0;
features = FOREACH grpd_order
{
    sorted = ORDER features BY idx;
    generate group as patientid, utils.bag_to_svmlight(sorted) as sparsefeature;
}

-- ***************************************************************************
-- Split into train and test set
-- labels is of the form (patientid, label) and contains all patientids followed by label of 1 for dead and 0 for alive
-- e.g. 1,1
--	2,0
--      3,1
-- ***************************************************************************

-- create it of the form (patientid, label) for dead and alive patients
deadlabel = foreach deadevents generate patientid, label;
alivelabel = foreach aliveevents generate patientid, label;
labels = UNION alivelabel, deadlabel;

--Generate sparsefeature vector relation
samples = JOIN features BY patientid, labels BY patientid;
samples = DISTINCT samples PARALLEL 1;
samples = ORDER samples BY $0;
samples = FOREACH samples GENERATE $3 AS label, $1 AS sparsefeature;

--TEST-6
STORE samples INTO 'samples' USING PigStorage(' ');

-- randomly split data for training and testing
DEFINE rand_gen RANDOM('8803');
samples = FOREACH samples GENERATE rand_gen() as assignmentkey, *;
SPLIT samples INTO testing IF assignmentkey <= 0.20, training OTHERWISE;
training = FOREACH training GENERATE $1..;
testing = FOREACH testing GENERATE $1..;

-- save training and tesing data
STORE testing INTO 'testing' USING PigStorage(' ');
STORE training INTO 'training' USING PigStorage(' ');
