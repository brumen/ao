// autocomplete feature of the ao
$("#js-origin-input").autocomplete({
    //source: "http://localhost/~brumen/cgi-bin/ao_auto_fill_origin.cgi"
    source: "cgi-bin/ao_auto_fill_origin.cgi"
});
$("#js-destination-input").autocomplete({
    source: "cgi-bin/ao_auto_fill_origin.cgi"
});
$("#airline-name").autocomplete({
    source: "cgi-bin/ao_auto_fill_airline.cgi"
});
