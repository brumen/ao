// dynamics of the AirOptions page 
//
function sel_all_flights() {
    // function that selects all flights
    source = document.getElementById("cb_sel_all");
    checkboxes = document.getElementsByName('flight_cb');
    for (var i=0, n=checkboxes.length; i<n; i++) {
	checkboxes[i].checked = source.checked;
    }
}


function create_cb_sel_all(parent_node, id_name, sel_all_fct) {
    // apends the checkbox to a parent_node with id_name and
    // the function sel_all_fct that acts when one clicks the button 
    var cb_sel_all = document.createElement("input");
    cb_sel_all.type    = "checkbox";
    cb_sel_all.style   = "margin-left: 5px;"
    cb_sel_all.id      = id_name;
    cb_sel_all.checked = true;
    cb_sel_all.onclick = sel_all_fct;

    parent_node.appendChild(cb_sel_all                    );
    parent_node.appendChild(document.createTextNode('All'));
}


function create_button_accordion_orig(element_to_append, button_text, div_id, button_id) {
    // creates the button and the accordion
    var outgoing_routes = document.createElement("button");
    outgoing_routes.className = "accordion";
    outgoing_routes.id = button_id; 
    outgoing_routes.appendChild(document.createTextNode(button_text)); 
    var outgoing_routes_div = document.createElement("div")
    outgoing_routes_div.className = "panel";
    outgoing_routes_div.id = div_id;
    element_to_append.appendChild(outgoing_routes);
    element_to_append.appendChild(outgoing_routes_div);
    return outgoing_routes_div;
}


function select_deselect_all_below(elt, button_id, checked_ind) {
    // selects or deselects all the indicators
    var div_used;
    var input_l;
    var idx, idx2;
    if (button_id == 'outgoing_all' || button_id == 'incoming_all') {  // handles departing flights or incoming flights 
	// get buttons & check the elts there
	if (button_id == 'outgoing_all')
	    div_used = document.getElementById('outgoing_flights');
	else
	    div_used = document.getElementById('incoming_flights');
	
	var buttons_l = div_used.getElementsByTagName('button');
	var buttons_l_len = buttons_l.length;
	for (idx = 0; idx<buttons_l_len; idx++) {
	    var input_sel_l = buttons_l[idx].getElementsByTagName('input');  // there is just one
	    var input_sel_l_len = input_sel_l.length;
	    for (idx2=0; idx2 < input_sel_l_len; idx2++) {
		var input_sel = input_sel_l[idx2];
		if (input_sel.checked != checked_ind)
		    input_sel.click();  // that has a handler 
	    }
	}
    } else { // handles date level and time_of_day level 
	div_used = document.getElementById(button_id + '_div');
	input_l = div_used.getElementsByTagName('input');
	var input_l_len = input_l.length;
	for (var idx=0; idx<input_l_len; idx ++)
	    if (input_l[idx].checked != checked_ind)
		input_l[idx].click();
    }
}


function create_button_accordion(element_to_append
				 , button_text
				 , div_id
				 , button_id) {
    // button_id: this is in the form of '2017-02-26' or '2017-02-26_night'
    // creates the button and the accordion in the lower buckets
    var outgoing_routes = document.createElement("button");
    // create a checkbox for this 
    var checkbox_elt = document.createElement("input");
    checkbox_elt.type = "checkbox";
    checkbox_elt.onclick = function() {
	var checked_ind = this.checked; 
	select_deselect_all_below(outgoing_routes, button_id, checked_ind);
    }
    checkbox_elt.checked = true;
    // create accordion 
    outgoing_routes.className = "accordion";
    outgoing_routes.id = button_id;
    outgoing_routes.appendChild(checkbox_elt);
    outgoing_routes.appendChild(document.createTextNode(button_text)); 
    var outgoing_routes_div = document.createElement("div")
    outgoing_routes_div.className = "panel";
    outgoing_routes_div.id = div_id;
    element_to_append.appendChild(outgoing_routes);
    element_to_append.appendChild(outgoing_routes_div);
    return outgoing_routes_div;
}


function date_to_string_2(date_i) {
    // converts the date in string format '20180415' to Date format. 
    // -1 in the month cause they start at 0 
    return new Date(parseInt(date_i.substring(0,4)), parseInt(date_i.substring(4,6))-1, parseInt(date_i.substring(6,8)));
}


function reorder_price_range(price_range, ow_ind) {
    // reorders price range for one-way or return flights
    // price_range_ow ... object of dates: prices
    // ow_ind ... one-way indicator 
    var arr_dates = [];
    var date_i;
    if (ow_ind) {
	for (date_i in price_range)
	    arr_dates.push(date_to_string_2(date_i));
	arr_dates.sort(function(a, b) {return a-b;});
    } else {  // return flights 
	for (date_i in price_range) {
	    var date_i_spl = date_i.split(' - ');  // array (beg. date, end date)
	    arr_dates.push([date_to_string_2(date_i_spl[0]), date_to_string_2(date_i_spl[1]) ] ) ;
	}
	arr_dates.sort(function(a,b) {return a[0] - b[0];});
    }
    return arr_dates;
}


function date_to_string(date_i) {
    // converts the date in Date format to a string format e.g. '20180301'
    var month = date_i.getMonth() + 1;
    var day   = date_i.getDate();
    var month_str, day_str;
    if (month < 10)
	month_str = "0" + String(month);
    else
	month_str = String(month);

    if (day < 10) 
	day_str = "0" + String(day);
    else
    	day_str = String(day);

    return String(date_i.getFullYear()) + month_str + day_str;
}

function date_to_string_slash(date_i) {
    // converts the date in Date format to the / format 
    return String(date_i.getMonth() + 1) + '/' + date_i.getDate() + '/' + String(date_i.getFullYear());
}

function present_price_ranges(price_range) {
    // writes the price ranges into the field where it displays them 
    //
    var one_way_ind = document.getElementById('js-one-way-input').checked;
    var options_display = document.getElementById('options-display');
    options_display.textContent = "";  // clean out 
    var price_range_ol = document.createElement("ol");
    options_display.appendChild(price_range_ol); 
    // sort dates
    var dates_sorted = reorder_price_range(price_range, one_way_ind);
    var option_range;
    for (var ord=0; ord < dates_sorted.length; ord ++) {
	if (one_way_ind)
	    option_range = date_to_string(dates_sorted[ord])
	else {
	    var out_option_range = date_to_string(dates_sorted[ord][0]);
	    var in_option_range  = date_to_string(dates_sorted[ord][1]);
	    option_range         = out_option_range + ' - ' + in_option_range;
	}
	var curr_val = price_range[option_range];
	var li_item = document.createElement("li");
	// constructing button element
	var button_elt = document.createElement("button");
	if (one_way_ind) 
	    button_elt.appendChild(document.createTextNode("Book departure until " + date_to_string_slash(dates_sorted[ord]) + ": " + curr_val + " USD"));
	else
	    button_elt.appendChild(document.createTextNode("Book departure until " + date_to_string_slash(dates_sorted[ord][0]) +
							   ", return until " + date_to_string_slash(dates_sorted[ord][1]) +
							   ": " + curr_val + " USD"));
	button_elt.id = option_range;
	button_elt.value = curr_val;
	button_elt.onclick = function() {var res = book_range(this); return false;};
	if (one_way_ind) 
	    button_elt.style.width = "100%"; // "270px";
	else
	    button_elt.style.width = "100%"; // "570px";
	// adding the li elements
	li_item.appendChild(button_elt);
	// adding li to the ol list 
	price_range_ol.appendChild(li_item);
    }
}


function display_results_init(response) {
    // displays the flights from the search query in the designated window
    // response is already JSON parsed 

    document.getElementById("flights-section").style = "";
    var flights_presented = document.getElementById("flights_presented");  // reserved space for flights
    var options_display = document.getElementById('options-display');  // reserved space for options 
    options_display.style = "background: white;";  // enable options 
    var price = response.price;
    if (!response.valid_inp) {  // are inputs valid 
	// delete children of options_display (if anything displayed before)
	while (options_display.firstChild) {
	    options_display.removeChild(options_display.firstChild);
	}
	// indicate that there is something wrong 
	flights_presented.textContent = "No flights found matching selected conditions.";
	return price;  // else continue w/ this
    }

    var flights       = response.flights;
    var reorg_flights = response.reorg_flights;
    var minmax        = response.minmax;  // minmax over subsets

    // present price ranges:
    present_price_ranges(response.price_range);
    flights_presented.textContent = "";  // clearing up document 
    
    // store data to localStorage
    localStorage.clear();
    localStorage.setItem("flights_found"     , "true"                                            );
    localStorage.setItem("option_start"      , document.getElementById("option-start-date").value);
    localStorage.setItem("option_end"        , document.getElementById("option-end-date").value  );
    localStorage.setItem("dep_start"         , document.getElementById("js-depart-input").value  );
    localStorage.setItem("dep_end"           , document.getElementById("js-return-input").value  );
    localStorage.setItem("carrier"           , document.getElementById("airline-name").value     );
    localStorage.setItem("strike"            , document.getElementById("ticket-price").value     );
    localStorage.setItem("flights_curr"      , JSON.stringify(flights)                           );
    localStorage.setItem("reorg_flights_curr", JSON.stringify(response.reorg_flights)                 );
    localStorage.setItem("minmax"            , JSON.stringify(minmax)                            );
    
    var origin_station = document.getElementById('js-origin-input').value;
    var dest_station   = document.getElementById('js-destination-input').value;
    
    // add All/None selector - THIS ONE WORKS 
    // create_cb_sel_all(flights_presented, 'cb_sel_all', sel_all_flights);
    if (response.return_ind == "one-way") {
	mm = minmax['min_max']
	create_button_accordion(flights_presented
				, "Departing flights: " + origin_station + " -> " + dest_station + ": from " + mm[0] + " USD to " + mm[1] + " USD."
				, "outgoing_flights", "outgoing_all" );
	reorganize_results_ow( flights
			       , reorg_flights
			       , minmax
			       , "outgoing_flights"
			       , "outgoing" );
    } else {
	mm_out = minmax[0]['min_max'];
	mm_in  = minmax[1]['min_max'];
	create_button_accordion(flights_presented
				, "Departing flights: " + origin_station + " -> " + dest_station + ": from " + mm_out[0] + " USD to " + mm_out[1] + " USD."
				, "outgoing_flights", "outgoing_all" );
	reorganize_results_ow(flights[0]
			      , reorg_flights[0]
			      , minmax[0]
			      , "outgoing_flights"
			      , "outgoing" );
	create_button_accordion(flights_presented
				, "Return flights: " + dest_station + " -> " + origin_station + ": from " + mm_in[0] + " USD to " + mm_in[1] + " USD."
				, "incoming_flights", "incoming_all" );
	reorganize_results_ow(flights[1]
			      , reorg_flights[1]
			      , minmax[1]
			      , "incoming_flights"
			      , "incoming" );
    }

    // accordion function onclick
    var acc = document.getElementsByClassName("accordion");
    for (var i = 0; i < acc.length; i++) {
	acc[i].onclick = function(){
	    this.classList.toggle("active");
	    this.nextElementSibling.classList.toggle("show");
	    return false;
        }
    }
    return price;
}


function reorganize_results_ow(flights
			       , reorg_flights
			       , minmax
			       , top_dom
			       , out_in_ind) {
    // displays the flights from the search query in the designaated window (from one-way)
    // top_dom: top object to which this is subordinated
    // minmax: information about the minimum and maximum flight 
    var nb_dates = reorg_flights.length;  // first element is the date
    var nb_flights = flights.length;
    var flights_presented = document.getElementById(top_dom);  // reserved space for flights
    // add All/None selector
    // create_cb_sel_all(flights_presented, 'cb_sel_all', sel_all_flights);
    
    // Dates ol
    var dates_selector = document.createElement("div");
    flights_presented.appendChild(dates_selector);
    
    for (var flight_date in reorg_flights) {  // iteration over dates 
	var curr_date = reorg_flights[flight_date];
	var mm_local = minmax[flight_date]; 
	li_dates = create_button_accordion(dates_selector
					   , flight_date + ": from " + mm_local['min_max'][0] + " USD to " + mm_local['min_max'][1] + " USD."
					   , flight_date + "_div"
					   , flight_date );
	
	var time_of_day_selector = document.createElement("ol");
	time_of_day_selector.id = "time_of_day_selectable"; 
	li_dates.appendChild(time_of_day_selector);

	for (var time_of_day in curr_date) {
	    // here are times of day 
	    var mm_local_2       = mm_local[time_of_day];

	    var li_time_of_day = document.createElement("li");
	    li_time_of_day.class = "ui-widget-content";
	    li_time_of_day.style = "list-style-type: none; border: 2px solid black; border-radius: 5px; margin-bottom: 5px; margin-right: 3px; margin-left:-20px;";
	    time_of_day_selector.appendChild(li_time_of_day);
	    
	    flight_selector = create_button_accordion(li_time_of_day
						      , time_of_day + ": from " + mm_local_2[0] + " USD to " + mm_local_2[1] + " USD."
						      , flight_date + '_' + time_of_day + '_div', flight_date + '_' + time_of_day );
	    li_time_of_day.appendChild(flight_selector);
	    display_flights_inner(flight_date
				  , curr_date[time_of_day]
				  , flight_selector
				  , out_in_ind); // display flights 
	}
    }
}


function remove_add_flight_from_reorg(reorg_flights_o, flight_date, flight_nb, remove_add_ind) {
    // given the object reorg_flights_o, it removes the flight from reorg_flights_o
    // doesnt return anything
    // remove_add_ind ... remove, add indicator (true - for adding, false - for removing)
    var curr_reorg = reorg_flights_o[flight_date]
    for (var flight_part_of_day in curr_reorg) {
	curr_part_of_day = curr_reorg[flight_part_of_day];
	for (var flight_time in curr_part_of_day) {
	    if (flight_time != 'min_max') {
		obj_curr = curr_part_of_day[flight_time];
		if (obj_curr[6] == flight_nb) {
		    // set the last part of the object to false
		    if (remove_add_ind) 
			curr_part_of_day[flight_time][7] = true;
		    else
			curr_part_of_day[flight_time][7] = false;
		    break;  // we can stop here
		}
	    }
	}
    }
}


function reconst_minmax_in_reorg(rof) {
    // reconstructs the minmax object 
    // rof ... reorg flights object (for outgoing/incoming flights only, not both)
    var min_max_o = Object()
    //var change_dates = rof.keys()
    var total_min = 1000000.;
    var total_max = 0.;
    for (var c_date in rof) { 
        min_max_o[c_date] = Object();
        var flights_by_daytime = rof[c_date]
        // flight_daytimes = rof[c_date].keys()
        var cd_min = 1000000.;
	var cd_max = 0.;
        for (var f_daytime in flights_by_daytime) {
            var flight_subset = flights_by_daytime[f_daytime]
	    //  now find minimum or maximum
            var min_subset = 1000000.;
	    var max_subset = 0.; 
            for (var d_date in flight_subset) {
		if (flight_subset[d_date][7]) {
		    if (flight_subset[d_date][5] < min_subset)
			min_subset = flight_subset[d_date][5];
                    if (flight_subset[d_date][5] >= max_subset)
			max_subset = flight_subset[d_date][5];
		}
	    }
            flight_subset['min_max'] = [min_subset, max_subset];
            min_max_o[c_date][f_daytime] = [min_subset, max_subset];
            if (min_subset < cd_min)
                cd_min = min_subset;
            if (max_subset >= cd_max)
                cd_max = max_subset;
	}
        min_max_o[c_date]['min_max'] = [cd_min, cd_max];
        if (total_min > cd_min)
            total_min = cd_min;
        if (total_max < cd_max)
            total_max = cd_max;
    }
    min_max_o['min_max'] = [total_min, total_max];
    return min_max_o;
}


function replace_text_in_button(elt, new_text) {
    // structure of nodes 
    var children = elt.childNodes;  // list of [checkbox, 'text']
    children[1].textContent = new_text;
}


function update_minmax_in_accordion(minmax_o, out_in_ind) {
    // updates the accordion with new min_maxes 
    if (out_in_ind == 'outgoing')
	var outgoing_level_button = document.getElementById('outgoing_all'); // highest button
    else
	var outgoing_level_button = document.getElementById('incoming_all'); // highest button

    var outgoing_level_txt = outgoing_level_button.textContent.split(':'); // we update just [2] one
    var mm_top = minmax_o['min_max'];
    var outgoing_level_new_txt;
    if (mm_top[0] == 1000000 && mm_top[1] == 0)
	outgoing_level_new_txt = outgoing_level_txt[0] + ':' + outgoing_level_txt[1] + ': (None selected)';
    else
	outgoing_level_new_txt = outgoing_level_txt[0] + ':' + outgoing_level_txt[1] + ': from ' + String(mm_top[0]) + ' USD to ' +
	String(mm_top[1]) + ' USD.';
    //outgoing_level_button.textContent = outgoing_level_new_txt;
    replace_text_in_button(outgoing_level_button, outgoing_level_new_txt);

    // descend to date level
    var mm_u;
    var date_button_txt_new;
    var tod_button_txt_new;
    for (flight_date in minmax_o) {
	if (flight_date != 'min_max') {  // dates considered
	    var date_button = document.getElementById(flight_date);
	    var date_button_txt = date_button.textContent.split(':');  // only 2 elements, replacing [1]
	    var mm_day = minmax_o[flight_date]['min_max'];
	    if (mm_day[0] == 1000000 && mm_day[1] == 0) {
		date_button_txt_new = date_button_txt[0] + ': (None selected)';
	    } else
		date_button_txt_new = date_button_txt[0] + ': from ' + String(mm_day[0]) + ' USD to ' + String(mm_day[1]) + ' USD.';
	    //date_button.textContent = date_button_txt_new;
	    replace_text_in_button(date_button, date_button_txt_new);
	    for (time_of_day in minmax_o[flight_date]) {  // handling time of day
		if (time_of_day != 'min_max') {
		    mm_u = minmax_o[flight_date][time_of_day];
		    var tod_button = document.getElementById(flight_date + '_' + time_of_day);
		    var tod_button_txt = tod_button.textContent.split(':');
		    if (mm_u[0] == 1000000 && mm_u[1] == 0)
			tod_button_txt_new = tod_button_txt[0] + ': (None selected)';
		    else
			tod_button_txt_new = tod_button_txt[0] + ': from ' + String(mm_u[0]) + ' USD to ' + String(mm_u[1]) + ' USD.';			
		    //tod_button.textContent = tod_button_txt_new;
		    replace_text_in_button(tod_button, tod_button_txt_new);
		}											
	    }
	}
    }
}


function update_minmax_upwards(elt) {
    // checkbox element was clicked - update the reorg_list and flights_list and minmax on localStorage 
    // remove/add  the flight to the selection
    var one_way_ind        = document.getElementById('js-one-way-input').checked;  // true if one-way
    var out_in_ind         = elt.name;  // 'outgoing' or 'incoming' 
    var date_flight        = elt.id.split(':');
    var date_used          = date_flight[0];
    var flight_used        = date_flight[1];
    var reorg_flights_curr = JSON.parse(localStorage.reorg_flights_curr);
    var minmax_old         = JSON.parse(localStorage.minmax);

    var reorg_flights_used_now, minmax_new;
    
    if (one_way_ind)
	remove_add_flight_from_reorg(reorg_flights_curr, date_used, flight_used, elt.checked);
    else {  // return flights 
	if (out_in_ind == 'outgoing')
	    reorg_flights_used_now = reorg_flights_curr[0];
	else
	    reorg_flights_used_now = reorg_flights_curr[1];
	
	remove_add_flight_from_reorg( reorg_flights_used_now
				      , date_used
				      , flight_used
				      , elt.checked );
    }
    localStorage.setItem('reorg_flights_curr'
			 , JSON.stringify(reorg_flights_curr) );  // updating the reorg elt. 
    if (one_way_ind) {
	minmax_new = reconst_minmax_in_reorg(reorg_flights_curr); // this is an object
	localStorage.setItem('minmax', JSON.stringify(minmax_new));  // updating the minmax elt.
    } else {  // return flight
	if (out_in_ind == 'outgoing') {
	    minmax_new = reconst_minmax_in_reorg(reorg_flights_curr[0]);
	    minmax_old[0] = minmax_new;
	} else {
	    minmax_new = reconst_minmax_in_reorg(reorg_flights_curr[1]);
	    minmax_old[1] = minmax_new;
	}
	localStorage.setItem('minmax', JSON.stringify(minmax_old));  // updating the minmax elt.
    }
    
    if (out_in_ind == 'outgoing') {
	if (one_way_ind) 
	    update_minmax_in_accordion(minmax_new, out_in_ind);
	else
	    update_minmax_in_accordion(minmax_old[0], out_in_ind);
    } else
	update_minmax_in_accordion(minmax_old[1], out_in_ind);  // out_in_ind == 'incoming', always for return
}


function display_flights_inner(flight_date, curr_time_of_day, flight_selector, out_in_ind) {
    // flights iteration 
    for (var curr_flight_date in curr_time_of_day) {
	if (curr_flight_date != 'min_max') {
	    var curr_flight = curr_time_of_day[curr_flight_date];
	    var checkbox_elt = document.createElement("input");
	    checkbox_elt.type    = "checkbox";
	    checkbox_elt.id      = flight_date + ":" + curr_flight[6];  // node id: flight_date : flight_id 
	    checkbox_elt.name    = out_in_ind;
	    checkbox_elt.checked = true;
	    checkbox_elt.onclick = function() {update_minmax_upwards(this);}
	    var li_flight = document.createElement("li");
	    li_flight.appendChild(checkbox_elt);
	    li_flight.appendChild(document.createTextNode(curr_flight[6] + ": " + curr_flight[2] + " - " + curr_flight[4] + " : " + curr_flight[5]));  // toFixed rounds to 2 decimal places
	    flight_selector.appendChild(li_flight);  // adding the flight
	}
    }
}


function reorganize_results_for_recompute(response) {
    // reorganizes results from the Python response 
    var obj = JSON.parse(response);
    present_price_ranges(obj.result.price_range);
    return obj.price; // this price is handled lower (not optimal)
}


function reorganize_results_inquiry(response) {
    // reorganizes results from the Python response 
    var obj = JSON.parse(response);
    var price = obj.price;
    return price; 
}


function get_ow_ret(elt_name) {
    // gets the return or one-way key from elt_name
    var trip_options = document.getElementsByName(elt_name);
    var return_ow;
    for(var i = 0; i < trip_options.length; i++){
	if(trip_options[i].checked) {
            return_ow = trip_options[i].value;
	}
    }
    return return_ow; // contains the 'return' or 'one-way' string 
}


function get_cabin_class(elt_name) {
    // gets the cabin class key from elt_name
    var cc_options = document.getElementsByName(elt_name)[0];
    var return_cc;  
    for(var i = 0; i < cc_options.length; i++){
	if(cc_options[i].selected) {
            return_cc = cc_options[i].value;
	}
    }
    return return_cc; // contains the 'return' or 'one-way' string 
}


function get_nb_people(elt_name) {
    // gets the cabin class key from elt_name
    var nb_people_options = document.getElementsByName(elt_name)[0];
    var return_nb_people;  
    for(var i = 0; i < nb_people_options.length; i++) {
	if (nb_people_options[i].selected) {
            return_nb_people = nb_people_options[i].value;
	}
    }
    return return_nb_people; // contains the number of people
}


function get_basic_info() {
    // constructs the object with the basic flight info 
    var return_ow = get_ow_ret("trip-type")

    return_obj = new FormData();
    return_obj['return_ow'     ] = return_ow;
    return_obj['cabin_class'   ] = get_cabin_class("cabin-class-type");
    return_obj['nb_people'     ] = get_nb_people("nb-people");
    return_obj['origin_place'  ] = document.getElementById("js-origin-input").value;
    return_obj['dest_place'    ] = document.getElementById("js-destination-input").value;
    return_obj['option_start'  ] = document.getElementById("option-start-date").value;
    return_obj['option_end'    ] = document.getElementById("option-end-date").value;
    return_obj['outbound_start'] = document.getElementById("js-depart-input").value;
    return_obj['outbound_end'  ] = document.getElementById("js-return-input").value;
    return_obj['ticket_price'  ] = document.getElementById("ticket-price").value;
    return_obj['airline_name'  ] = document.getElementById("airline-name").value;

    if (return_ow == 'return') {
	// get the return data
	return_obj['option_ret_start'  ] = document.getElementById("option-start-date-return").value;
	return_obj['option_ret_end'    ] = document.getElementById("option-end-date-return").value;
	return_obj['outbound_start_ret'] = document.getElementById("js-depart-input-return").value;
	return_obj['outbound_end_ret'  ] = document.getElementById("js-return-input-return").value;
    }

    return return_obj;
}


function construct_get_req_string(basic_info_obj) {
    // constructs the request string from the FormData object basic_info_obj 
    var return_ow = basic_info_obj['return_ow'];

    var get_string = "?origin_place="   + basic_info_obj['origin_place']
	           + "&dest_place="     + basic_info_obj['dest_place']
	           + "&option_start="   + basic_info_obj['option_start']
	           + "&option_end="     + basic_info_obj['option_end']
	           + "&outbound_start=" + basic_info_obj['outbound_start']
	           + "&outbound_end="   + basic_info_obj['outbound_end']
	           + "&ticket_price="   + basic_info_obj['ticket_price']
	           + "&airline_name="   + basic_info_obj['airline_name']
	           + "&return_ow="      + return_ow
	           + "&cabin_class="    + basic_info_obj['cabin_class']
	           + "&nb_people="      + basic_info_obj['nb_people'];

    if (return_ow == 'return') {
	get_string += "&option_ret_start=" + basic_info_obj['option_ret_start']
	            + "&option_ret_end=" + basic_info_obj['option_ret_end']
	            + "&outbound_start_ret=" + basic_info_obj['outbound_start_ret']
	            + "&outbound_end_ret=" + basic_info_obj['outbound_end_ret'];
    }

    return get_string;
}


function change_button_to_searching( btn
				     , replace_txt
				     , left_or_after_ind ) {
    // change the button to a different class 
    btn.textContent = "";  // remove all text 
    if (left_or_after_ind == 'left')
	btn.className = 'js-search-button wc-button-large-left btn btn-default btn-md';
    else
	btn.className = 'js-search-button wc-button-large-left-after btn btn-default btn-md';
    var i_elt = document.createElement('i');
    i_elt.className = 'fa fa-circle-o-notch fa-spin';
    i_elt.id        = 'spinning_btn';
    btn.appendChild(i_elt);
    btn.appendChild(document.createTextNode(replace_txt));
}


function change_button_back( btn
			     , replace_txt
			     , left_or_after_ind) {

    // left_or_after indicator is whether the button receives left or left_after class 
    if (left_or_after_ind == 'left')
	btn.className = 'js-search-button wc-button-large-left';
    else
	btn.className = 'js-search-button wc-button-large-left-after';
    var i_elt = document.getElementById('spinning_btn');
    i_elt.parentNode.removeChild(i_elt);
    btn.textContent = replace_txt;
}


function compute_option_from_init() {
    // computes the option value and displays flights 
    change_button_to_searching(document.getElementById('find_flights_button')
			       , '  Searching...', 'left');
    handle_computation(construct_get_req_string(get_basic_info()));
}


function handle_computation(get_string) {
    // Handles the computation of the option w/ server side events 
    
    // clean notification_messages
    var nm = document.getElementById("notification_messages");
    while (nm.hasChildNodes())
	nm.removeChild(nm.lastChild);
    var fp = document.getElementById("flights_presented")
    while (fp.hasChildNodes())
	fp.removeChild(fp.lastChild);
    // delete options-display
    var od = document.getElementById("options-display")
    while (od.hasChildNodes())
	od.removeChild(od.lastChild);

    // display the notification area 
    document.getElementById("notification_messages").style = "";

    var eventSource = new EventSource("myapp/compute_option" + get_string );
    eventSource.onmessage = function (server_message) { // handling response from server
	// function handles the return messages from server
	// involving the option computation 
	var data = JSON.parse(server_message.data);
    
	if (data.finished)  // display elements 
	{   // success logic - data object with fields as defined 
	    // close the notification messages
	    document.getElementById("notification_messages").style = "display:none;";
 	    document.getElementById("option_price_frame").value = display_results_init(data.result);
 	    change_button_back(document.getElementById('find_flights_button')
			       , 'Find flights', 'left');
 	    // change the type of checkbox accorions 
 	    $('.accordion input[type="checkbox"]').click(function(e) {
 		e.stopPropagation();
 	    });

	    eventSource.close(); // close the SSE stream

	} else {  // not finished, display the message in the result 
 	    // showing the style 
 	    document.getElementById("flights-section").style = "";  // display this section 
 	    document.getElementById("notification_messages")
	        .appendChild(document.createElement("p")
			     .appendChild(document.createTextNode(data.result)))
 	    document.getElementById("notification_messages").appendChild(document.createElement("br"));
	}
    }

    eventSource.onerror = function (server_message) {
	var data = JSON.parse(server_message.data);
	console.log(data);  // display error in console
	document.getElementById("notification_messages").style = "";
	document.getElementById("notification_messages").appendChild(document.createTextNode('Something went wrong. Please try again.'));
	eventSource.close();
    }
}


function recompute_option_post() {
    // recompute option but with a post command 
    if (localStorage.flights_found != "true")  // this guarantees that flights_selected exists
	return;
    var info_o = get_basic_info();
    info_o['flights_selected'] = localStorage.reorg_flights_curr; // in string format
    info_o['price'] = document.getElementById('option_price_frame').value; // previous price (not sure why needed) 
    var recomp_button = document.getElementById('recompute_button');
    change_button_to_searching(recomp_button, '  Working...', 'left-after');
    var req = new XMLHttpRequest();
    req.onload = function() {
    //req.onreadystatechange = function() {
	var new_price = reorganize_results_for_recompute(req.responseText);
	if (!(new_price == '-1'))
	    document.getElementById("option_price_frame").value = new_price;
	change_button_back(recomp_button, 'Recompute', 'left-after');
    }
    req.open("POST", "myapp/recompute_option", true);
    req.setRequestHeader("Content-type", "application/json");
    req.send(JSON.stringify(info_o));
}


function open_inquiry() {
    // open a small inquiry window for text entry & sending
    var inquiry_form = document.getElementById("inquiry_form");
    inquiry_form.style = "";
    //$('inquiry_form').css('display','');
}


function close_inquiry() {
    // closes the inquiry window
    $('#inquiry_form').hide();
}


function send_inquiry() {
    // sends the inquiry 
    var info_o             = get_basic_info();
    info_o['message_text'] = document.getElementById("inquiry_text_form").value;
    info_o['email_text'  ] = document.getElementById("inquiry_email").value;

    if (typeof localStorage.reorg_flights_curr !== "undefined")
	info_o['flights_sel'] = JSON.parse(localStorage.reorg_flights_curr)
    else
	info_o['flights_sel'] = "undefined";
    
    var req = new XMLHttpRequest();
    req.open("POST", "myapp/write_inquiry", true);
    req.setRequestHeader("Content-type", "application/json");
    req.send(JSON.stringify(info_o));
    
    // remove the inquiry field and button
    var inquiry_form = document.getElementById("inquiry_form");
    inquiry_form.style = "display: none;"
}


function add_fields_for_return() {
    // adds the fields in the document required for return
    // used in radio button onclick event
    all_return = document.getElementById("all_return_options")
    all_return.style = ""; // "display: none;"
}

function remove_fields_for_return() {
    // hides button for the return flights
    // used in radio button onclick event 
    all_return = document.getElementById("all_return_options")
    all_return.style = "display: none;"
}


function swap_locations_fct() {
    // swaps origin and destination location 
    var origin = document.getElementById("js-origin-input");
    var dest = document.getElementById("js-destination-input");
    var dest_tmp = dest.value;
    dest.value = origin.value;
    origin.value = dest_tmp;    
}


function show_details() {
    var page_part = document.getElementById("all_return_options_2");
    var option_price_part = document.getElementById("option_price_all");
    var book_details_button = document.getElementById("book_details_button");
    
    var chbox_elt = document.getElementById("js-details-input");
    if (chbox_elt.checked) {
	page_part.style               = "";
	option_price_part.style       = "";
	book_details_button.style     = "";
	book_details_button.className = "js-search-button wc-button-large-left-after";
    } else {
	page_part.style           = "display: none";
	option_price_part.style   = "display: none";
	book_details_button.style = "display: none";
    }
}


function copy_basics(btn_elt, opt_val) {
    // copies the relevant elements from the page to a form, which can be sent for inquiry 
    form_data = get_basic_info();  // returns the form w/ basic data 
    form_data['curr_sel_option_value'] = opt_val;
    form_data['curr_sel_flights_sel_final'] = localStorage.reorg_flights_curr;
    // option end dates 
    if (form_data['return_ow'] == 'return') {  // return flight 
	var dep_ret_split = btn_elt.id.split('-');
	form_data['opt_end_dep_final'] = dep_ret_split[0].substring(0, dep_ret_split[0].length -1);
	form_data['opt_end_ret_final'] = dep_ret_split[1].substring(1, dep_ret_split[1].length);
    } else  // one-way flight 
	form_data['opt_end_dep_final'] = btn_elt.id;
    
    return form_data;
}


function book_range(btn_elt) {
    // books the range for this particular element
    form_data = copy_basics(btn_elt, btn_elt.value);  // copies the basic elements 

    // scrolls to the element
    // document.getElementById('payment_form').scrollIntoView();
    open_inquiry(); 
    document.getElementById('options-display').scrollIntoView();
    return false;
}


// verifies if the origin or destination or airline is changed 
function verify_origin_dest(orig_dest) {
    if (orig_dest == 'origin') 
	var origin_inp = document.getElementById('js-origin-input');
    else if (orig_dest == 'dest')
	var origin_inp = document.getElementById('js-destination-input');
    else if (orig_dest == 'airline')
	var origin_inp = document.getElementById('airline-name');
	
    var req = new XMLHttpRequest();
    var origin_name = origin_inp.value;
    // TODO: CHECK HERE WHICH IS MORE APPROPRIATE
    // req.onreadystatechange = function() {
    req.onload = function() {
	close_flights_booking();
	var is_valid = JSON.parse(req.responseText).found;
	if (is_valid)
	    origin_inp.style.color = "black";
	else
	    origin_inp.style.color = "red";
    }
    if (orig_dest == 'airline')
	req.open("GET", "myapp/verify_airline" + encodeURI("?airline=" + origin_name), true);
    else
	req.open("GET", "myapp/verify_origin" + encodeURI("?origin=" + origin_name), true);

    req.send();
}


function verify_origin() {
    verify_origin_dest('origin');
}


function verify_dest() {
    verify_origin_dest('dest');
}


function verify_airline() {
    verify_origin_dest('airline')
}


function close_flights_booking() {
    // closes flights and booking if any of the relevant inputs change 

    localStorage.flights_found = "false";  // so that the flights are reset

    var options_display = document.getElementById('options-display');
    // TODO: To remove after it works. 
    // var flights_presented = document.getElementById('flights-section');
    // flights_presented.style = "display: none;";
    $('#flights-section').hide(); //
    // delete children of options_display (if anything displayed before
    options_display.style = "display: none;";
    while (options_display.firstChild) {
	options_display.removeChild(options_display.firstChild);
    }
}


function populate_carriers() {
    // populates carriers from the database

    var req = new XMLHttpRequest();
    req.onload = function() {
	close_flights_booking(); // resets the values 
	var resp_parsed = JSON.parse(req.responseText);
	var is_valid = resp_parsed.is_valid;
	var list_carriers = resp_parsed.list_carriers;

	if (is_valid) {
	    $("#airline-name").autocomplete({
		source: list_carriers
	    }); 
	}
    }
    req.open("GET", "myapp/find_relevant_carriers" + encodeURI("?origin=" + document.getElementById('js-origin-input').value +
							       "&dest="   + document.getElementById('js-destination-input').value), true);
    req.send();
}


function max_dates(d1, d2) {
    // takes the maximum of 2 dates
    if (d2 - d1 >= 0)
	return d2;
    else
	return d1;
}


function max_dates_p1(d1, d2) {
    // max (d1, d2) + 1 day 
    if (d1.getTime() >= d2.getTime()) {
	rd = new Date(d1);
	rd.setDate(rd.getDate() + 1);  // next day
	return rd;
    } else
	return d2;
}


function convert_date_to_mm_dd_yyyy(d) {
    var mm = d.getMonth() + 1; // getMonth() is zero-based
    var dd = d.getDate();
    return [d.getFullYear(),
	    (mm > 9 ? '' : '0') + mm,
	    (dd > 9 ? '' : '0') + dd
           ].join('');
}


function departure_start_change() {
    // update dates if js-depart-input changes
    close_flights_booking();  // erase found flights 

    var dep_start = $("#js-depart-input");
    var dep_start_date = dep_start.datepicker('getDate');
    var dep_end = $("#js-return-input");
    var dep_end_date = dep_end.datepicker('getDate');
    var ret_start = $("#js-depart-input-return");
    var ret_start_date = ret_start.datepicker('getDate');
    var ret_end = $("#js-return-input-return");
    var ret_end_date = ret_end.datepicker('getDate');
 
    // updating departure_end date 
    var new_dep_end_date = new Date(dep_start_date);
    new_dep_end_date.setDate(max_dates(new_dep_end_date, dep_end_date).getDate());
    dep_end.datepicker("setDate", new_dep_end_date);

    // return start date 
    var new_ret_start_date = new Date(max_dates_p1(new_dep_end_date, ret_start_date));
    ret_start.datepicker("setDate", new_ret_start_date);
    
    // return end date
    var new_ret_end_date = new Date(ret_end_date);
    ret_end.datepicker("setDate", max_dates(new_ret_end_date, new_ret_start_date));

}
