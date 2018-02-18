#!/bin/bash



SELECT f.reg_id reg_id, fid.carrier carrier, fid.orig orig, fid.dest dest, f.flight_id flight_id,
             drift(td(f.as_of), f.price) drift, 
             vol_1(td(f.as_of), f.price) vol_1, 
             vol_2(td(f.as_of), f.price) vol_2, 
             SUM(f.price) price_sum,
             COUNT(f.as_of) nb_obs_raw,
             counti(td(f.as_of)) nb_obs
      FROM flights_ord f, flight_ids fid
      WHERE f.flight_id = fid.flight_id AND f.flight_id = 3222
      GROUP BY f.reg_id, fid.carrier, fid.orig, fid.dest, f.flight_id;

date
for y in {1988..2013}
do
    sql="select yeard, count(*) from ontime where yeard=$y"
    mysql -vvv ontime -e "$sql" &>par_sql1/$y.log &
done
wait
date
