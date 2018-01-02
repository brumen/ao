/* computes partial drift and vol for a date_l and price_l */
/* compile with: gcc $(mysql-config --cflags) -shared -fPIC -o comp_pdv.so comp_pdv.c */
/* CREATE AGGREGATE FUNCTION drift RETURNS REAL SONAME "comp_pdv.so"; */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>   
#include <vector>
#include <algorithm>

#include <mysql.h>
#include <mysql/my_global.h>
/* #include <mysql/my_sys.h> */
#include <mysql/m_string.h>  /* To get strmov() */
#include <mysql/m_ctype.h>

typedef unsigned long long ulonglong;
typedef long long longlong;
static pthread_mutex_t LOCK_hostname;


/*
  drift/vol computation 
  ** Syntax for the new aggregate commands are:
  ** create aggregate function <function_name> returns {string|real|integer}
  **		  soname <name_of_shared_library>
  **
  ** Syntax for avgcost: avgcost( t.time (secs (INT), t.price )
  **	with t.quantity=integer, t.price=double
  ** (this example is provided by Andreas F. Bobak <bobak@relog.ch>)
  */

typedef struct {
  ulonglong ts;
  double price;
} ts_price;

typedef std::vector<ts_price> tsp_vec;

struct by_ts { 
  bool operator()(ts_price const &a, ts_price const &b) { 
    return a.ts < b.ts;
  }
};


/* exports them as C functions */
C_MODE_START;
my_bool drift_init(UDF_INIT* initid, UDF_ARGS* args, char* message);
my_bool counti_init(UDF_INIT* initid, UDF_ARGS* args, char* message);
my_bool vol_1_init(UDF_INIT* initid, UDF_ARGS* args, char* message);
my_bool vol_2_init(UDF_INIT* initid, UDF_ARGS* args, char* message);
void drift_deinit(UDF_INIT* initid);
void vol_1_deinit(UDF_INIT* initid);
void vol_2_deinit(UDF_INIT* initid);
void counti_deinit(UDF_INIT* initid);
void drift_clear(UDF_INIT* initid, char* is_null MY_ATTRIBUTE((unused)),
		   char* message MY_ATTRIBUTE((unused)));
void vol_1_clear(UDF_INIT* initid, char* is_null MY_ATTRIBUTE((unused)),
		   char* message MY_ATTRIBUTE((unused)));
void vol_2_clear(UDF_INIT* initid, char* is_null MY_ATTRIBUTE((unused)),
		   char* message MY_ATTRIBUTE((unused)));
void counti_clear(UDF_INIT* initid, char* is_null MY_ATTRIBUTE((unused)),
		    char* message MY_ATTRIBUTE((unused)));
void drift_add(UDF_INIT* initid, UDF_ARGS* args,
		 char* is_null MY_ATTRIBUTE((unused)),
		 char* message MY_ATTRIBUTE((unused)));
void vol_1_add(UDF_INIT* initid, UDF_ARGS* args,
		 char* is_null MY_ATTRIBUTE((unused)),
		 char* message MY_ATTRIBUTE((unused)));
void vol_2_add(UDF_INIT* initid, UDF_ARGS* args,
		 char* is_null MY_ATTRIBUTE((unused)),
		 char* message MY_ATTRIBUTE((unused)));
void counti_add(UDF_INIT* initid, UDF_ARGS* args,
		  char* is_null MY_ATTRIBUTE((unused)),
		  char* message MY_ATTRIBUTE((unused)));
double drift(UDF_INIT* initid, UDF_ARGS* args MY_ATTRIBUTE((unused)),
	       char* is_null, char* error MY_ATTRIBUTE((unused)));
double vol_1(UDF_INIT* initid, UDF_ARGS* args MY_ATTRIBUTE((unused)),
	       char* is_null, char* error MY_ATTRIBUTE((unused)));
double vol_2(UDF_INIT* initid, UDF_ARGS* args MY_ATTRIBUTE((unused)),
	       char* is_null, char* error MY_ATTRIBUTE((unused)));
ulonglong counti(UDF_INIT* initid, UDF_ARGS* args MY_ATTRIBUTE((unused)),
		 char* is_null, char* error MY_ATTRIBUTE((unused)));
C_MODE_END;


/*
** Drift Aggregate Function.
*/
my_bool drift_init(UDF_INIT* initid, UDF_ARGS* args, char* message) {
  tsp_vec* data;
  
  if (args->arg_count != 2) {
    strcpy(message, "wrong number of arguments: DRIFT/VOL_1/VOL_2 requires two arguments");
    return 1;
  }
  if ((args->arg_type[0] != INT_RESULT) || (args->arg_type[1] != REAL_RESULT)) {
    strcpy(message, "wrong argument type: DRIFT/VOL_1/VOL_2() requires an INT and a REAL");
    return 1;
  }

  initid->maybe_null	= 0;		/* The result may be null */
  initid->decimals	= 4;		/* We want 4 decimals in the result */
  initid->max_length	= 20;		/* 6 digits + . + 10 decimals */

  if (!( data = new tsp_vec)) {
    my_stpcpy(message, "Couldn't allocate memory");
    return 1;
  }

  initid->ptr = (char*) data;
  return 0;
}

my_bool counti_init(UDF_INIT* initid, UDF_ARGS* args, char* message) {
  tsp_vec *data;
  if (args->arg_count != 1) {
    strcpy(message, "wrong number of arguments: COUNTI() requires one argument");
    return 1;
  }

  if (args->arg_type[0] != INT_RESULT) {
    strcpy(message, "wrong argument type: counti() requires an INT and a int");
    return 1;
  }

  initid->maybe_null	= 0;		/* The result may be null */
  initid->decimals	= 0;		/* We want 4 decimals in the result */
  /* initid->max_length	= 20; */		/* 6 digits + . + 10 decimals  */

  if (!(data = new tsp_vec)) {
    my_stpcpy(message, "Couldn't allocate memory");
    return 1;
  }

  initid->ptr = (char*)data;
  return 0;
}


my_bool vol_1_init(UDF_INIT* initid, UDF_ARGS* args, char* message) {
  return drift_init(initid, args, message);
}
my_bool vol_2_init(UDF_INIT* initid, UDF_ARGS* args, char* message) {
  return drift_init(initid, args, message);
}

void drift_deinit(UDF_INIT* initid) {
  void *void_ptr= initid->ptr;
  tsp_vec *data=  (tsp_vec*) void_ptr;
  free(data);
}

void vol_1_deinit(UDF_INIT* initid) {
  drift_deinit(initid);
}
void vol_2_deinit(UDF_INIT* initid) {
  drift_deinit(initid);
}
void counti_deinit(UDF_INIT* initid) {
  drift_deinit(initid);
}


/* This is only for MySQL 4.0 compability */
/* void avgcost_reset(UDF_INIT* initid, UDF_ARGS* args, char* is_null, char* message) */
/* { */
/*   avgcost_clear(initid, is_null, message); */
/*   avgcost_add(initid, args, is_null, message); */
/* } */

/* This is needed to get things to work in MySQL 4.1.1 and above */
void drift_clear(UDF_INIT* initid, char* is_null MY_ATTRIBUTE((unused)),
		 char* message MY_ATTRIBUTE((unused))) {
  tsp_vec* data = (tsp_vec*) initid->ptr;
  data->clear();
}

void vol_1_clear(UDF_INIT* initid, char* is_null MY_ATTRIBUTE((unused)),
		 char* message MY_ATTRIBUTE((unused))) {
  drift_clear(initid, is_null, message);
}
void vol_2_clear(UDF_INIT* initid, char* is_null MY_ATTRIBUTE((unused)),
		 char* message MY_ATTRIBUTE((unused))) {
  drift_clear(initid, is_null, message);
}
void counti_clear(UDF_INIT* initid, char* is_null MY_ATTRIBUTE((unused)),
		  char* message MY_ATTRIBUTE((unused))) {
  drift_clear(initid, is_null, message);
}


void drift_add(UDF_INIT* initid, UDF_ARGS* args,
	       char* is_null MY_ATTRIBUTE((unused)),
	       char* message MY_ATTRIBUTE((unused))) {

  tsp_vec* data = (tsp_vec*) initid->ptr;
  ulonglong new_secs = *((ulonglong*) args->args[0]);
  double new_price = *((double*) args->args[1]);
  ts_price tsp_new = {.ts = new_secs, .price = new_price};
  /* 315... is 365*86400 */
  data->push_back(tsp_new);
}

void vol_1_add(UDF_INIT* initid, UDF_ARGS* args,
	       char* is_null MY_ATTRIBUTE((unused)),
	       char* message MY_ATTRIBUTE((unused))) {
  drift_add(initid, args, is_null, message);
}

void vol_2_add(UDF_INIT* initid, UDF_ARGS* args,
	       char* is_null MY_ATTRIBUTE((unused)),
	       char* message MY_ATTRIBUTE((unused))) {
  drift_add(initid, args, is_null, message);
}

void counti_add(UDF_INIT* initid, UDF_ARGS* args,
		char* is_null MY_ATTRIBUTE((unused)),
		char* message MY_ATTRIBUTE((unused))) {
  tsp_vec* data = (tsp_vec*) initid->ptr;
  ulonglong new_secs = *((ulonglong*) args->args[0]);
  ts_price tsp_new = {.ts = new_secs, .price = 0.};
  data->push_back(tsp_new);
}


double drift(UDF_INIT* initid, UDF_ARGS* args MY_ATTRIBUTE((unused)),
	     char* is_null, char* error MY_ATTRIBUTE((unused))) {

  tsp_vec* data = (tsp_vec*) initid->ptr;
  *is_null = 0;  /* we are fine CHECK THIS*/
  /* sort prices & ts according to ts */
  std::sort(data->begin(), data->end(), by_ts());
  /* compute drift over sorted ts_vec */
  std::vector<ts_price>::iterator tsp_iter_prev, tsp_iter_next;
  double total_drift = 0.;
  double prev_price, next_price;
  ulonglong prev_secs, next_secs;

  if (data->size() > 1)
    for (tsp_iter_prev=data->begin(),
	   tsp_iter_next = data->begin()+1;
	 tsp_iter_prev != data->end()-1, tsp_iter_next != data->end();
	 tsp_iter_prev++, tsp_iter_next++) {
      prev_secs = tsp_iter_prev->ts;
      prev_price = tsp_iter_prev->price;
      next_secs = tsp_iter_next->ts;
      next_price = tsp_iter_next->price;
      if (next_secs > prev_secs)
	total_drift += (next_price - prev_price) / ((double)(next_secs - prev_secs)/31536000);
    }
  return total_drift;
}

double vol_1(UDF_INIT* initid, UDF_ARGS* args MY_ATTRIBUTE((unused)),
	     char* is_null, char* error MY_ATTRIBUTE((unused))) {

  tsp_vec* data = (tsp_vec*) initid->ptr;
  *is_null = 0;  /* we are fine CHECK THIS*/
  /* sort prices & ts according to ts */
  std::sort(data->begin(), data->end(), by_ts());
  /* compute drift over sorted ts_vec */
  std::vector<ts_price>::iterator tsp_iter_prev, tsp_iter_next;
  double total_drift = 0.;
  double prev_price, next_price;
  ulonglong prev_secs, next_secs;
  double price_diff, time_diff;
  
  if (data->size() > 1)
    for (tsp_iter_prev=data->begin(),
	   tsp_iter_next = data->begin()+1;
	 tsp_iter_prev != data->end()-1, tsp_iter_next != data->end();
	 tsp_iter_prev++, tsp_iter_next++) {
      prev_secs = tsp_iter_prev->ts;
      prev_price = tsp_iter_prev->price;
      next_secs = tsp_iter_next->ts;
      next_price = tsp_iter_next->price;
      if (next_secs > prev_secs) {
	price_diff = next_price - prev_price;
	time_diff = (double)(next_secs - prev_secs)/31536000.;
	/* fprintf(stderr, "CCC1 %g %g\n", price_diff, time_diff); */
	/* fflush(stderr); */
	total_drift += price_diff * price_diff / time_diff;
      }
    }
  return total_drift;
}
double vol_2(UDF_INIT* initid, UDF_ARGS* args MY_ATTRIBUTE((unused)),
	     char* is_null, char* error MY_ATTRIBUTE((unused))) {

  tsp_vec* data = (tsp_vec*) initid->ptr;
  *is_null = 0;  /* we are fine CHECK THIS*/
  /* sort prices & ts according to ts */
  std::sort(data->begin(), data->end(), by_ts());
  /* compute drift over sorted ts_vec */
  std::vector<ts_price>::iterator tsp_iter_prev, tsp_iter_next;
  double total_drift = 0.;
  double prev_price, next_price;
  ulonglong prev_secs, next_secs;
  double price_diff, time_diff;
  
  if (data->size() > 1)
    for (tsp_iter_prev=data->begin(),
	   tsp_iter_next = data->begin()+1;
	 tsp_iter_prev != data->end()-1, tsp_iter_next != data->end();
	 tsp_iter_prev++, tsp_iter_next++) {
      prev_secs = tsp_iter_prev->ts;
      prev_price = tsp_iter_prev->price;
      next_secs = tsp_iter_next->ts;
      next_price = tsp_iter_next->price;
      if (next_secs > prev_secs) {
	price_diff = next_price - prev_price;
	time_diff = (double)(next_secs - prev_secs)/31536000.;
	/* fprintf(stderr, "CCC2 %g %g\n", price_diff, time_diff); */
	/* fflush(stderr); */
	total_drift += price_diff / sqrt(time_diff);
      }
    }
  return total_drift;
}

ulonglong counti(UDF_INIT* initid, UDF_ARGS* args MY_ATTRIBUTE((unused)),
		 char* is_null, char* error MY_ATTRIBUTE((unused))) {
  tsp_vec* data = (tsp_vec*) initid->ptr;
  *is_null = 0;  /* we are fine CHECK THIS*/
  /* sort prices & ts according to ts */
  std::sort(data->begin(), data->end(), by_ts());
  /* compute drift over sorted ts_vec */
  std::vector<ts_price>::iterator tsp_iter_prev, tsp_iter_next;
  ulonglong total_count = 1;
  ulonglong prev_secs, next_secs;
  
  if (data->size() > 1)
    for (tsp_iter_prev=data->begin(),
	   tsp_iter_next = data->begin()+1;
	 tsp_iter_prev != data->end()-1, tsp_iter_next != data->end();
	 tsp_iter_prev++, tsp_iter_next++) {
      prev_secs = tsp_iter_prev->ts;
      next_secs = tsp_iter_next->ts;
      if (next_secs > prev_secs)
	total_count += 1;
    }
  return total_count;
}
