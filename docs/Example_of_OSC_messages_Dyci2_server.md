#Example of OSC messages to / from Dicy2 server 

This example corresponds to `tab 2` in tutorial `2_Chaining_Agents.maxpat` in [Dicy2 for Max](https://forum.ircam.fr/projects/detail/dicy2/)

## Launch server

Launch `dicy2_server.app` with OSC ports as arguments
*NB: this applies of course to `dicy2_server.app` or to a server started from the Python code!*

FROM_SERVER: /server initialized
FROM_SERVER: /server status bang *... (then looped)*


## Create two agents

### Create first agent (labels=abstract musical gesture types, contents=sequences of vocal playing modes)

#### Create first agent "Agent_Form"

TO_SERVER: /server create_agent /Agent_Form listlabel

FROM_SERVER: /server info "Created new agent at '/Agent_Form'"
FROM_SERVER: /server create_agent /Agent_Form
FROM_SERVER: /Agent_Form initialized
FROM_SERVER: /Agent_Form status bang *... (then looped)*
 

#### Setting some parameters for first agent (in particular the max_continuity parameters)

TO_SERVER: /Agent_Form set_control_parameter generator::prospector::\_navigator::max_continuity 6
TO_SERVER: /Agent_Form set_control_parameter generator::force_output 1
TO_SERVER: /Agent_Form clear listlabel

FROM_SERVER: /Agent_Form clear ListLabel ... (+ list of default paramaters)


#### Learning a sequence in first agent (labels=abstract musical gesture types, contents=sequences of vocal playing modes)

TO_SERVER: /Agent_Form learn_event listlabel Gesture_type_1 "Agilita Breath Ribatutto Silence Staccato Tenuto"
TO_SERVER: /Agent_Form learn_event listlabel Gesture_type_2 "Staccato Staccato Staccato"
(... send other couples {label} {content})

FROM_SERVER: /Agent_Form new_event_learned "Agilita Breath Ribatutto Silence Staccato Tenuto"
FROM_SERVER: /Agent_Form info "Learned event 'Agilita Breath Ribatutto Silence Staccato Tenuto'"
FROM_SERVER: /Agent_Form new_event_learned "Staccato Staccato Staccato"
FROM_SERVER: /Agent_Form info "Learned event 'Staccato Staccato Staccato'"
(... feedback other couples {label} {content})

### Create second agent (labels=vocal playing mode, contents=corresponding markers in an audio file)

#### Create second agent "Agent_Audio"

TO_SERVER: /server create_agent /Agent_Audio listlabel

FROM_SERVER: /server info "Created new agent at '/Agent_Audio'"
FROM_SERVER: /server create_agent /Agent_Audio
FROM_SERVER: /Agent_Audio initialized
FROM_SERVER: /Agent_Audio status bang *... (then looped)*

#### Setting some parameters for second agent (in particular the max_continuity parameters)

TO_SERVER: /Agent_Audio set_control_parameter generator::prospector::\_navigator::max_continuity 6
TO_SERVER: /Agent_Audio set_control_parameter generator::force_output 1

FROM_SERVER: /Agent_Audio clear ListLabel ... (+ list of default paramaters)

#### Learning a sequence in first agent (labels=vocal playing mode, contents=corresponding markers in an audio file)

TO_SERVER: /Agent_Audio clear listlabel
TO_SERVER: /Agent_Audio learn_event listlabel Tenuto "0 1876"
TO_SERVER: /Agent_Audio learn_event listlabel Agilita "1876 2398"
(... send other couples {label} {content})

FROM_SERVER: /Agent_Audio new_event_learned "0 1876"
FROM_SERVER: /Agent_Audio info "Learned event '0 1876'"
FROM_SERVER: /Agent_Audio new_event_learned "1876 2398"
FROM_SERVER: /Agent_Audio info "Learned event '1876 2398'"
(... feedback other couples {label} {content})


## Send queries to the agents

### First agent "/Agent_Form"

#### Send a "free" query to the first agent "/Agent_Form"

*A "free" query takes advantage of the model built on the sequence to generate a new sequence without any constraint than the `maxcontinuity` value defined earlier*

TO_SERVER: /Agent_Form query query0 0 relative 10 

__Note the syntax of a "free" query :
`{name_agent} query {name_query} {start_date} {relative or absolute} {number_of_events_to_generate}`__

#### Receiving and parsing the result of the query

FROM_SERVER: /Agent_Form server_received_query query0
FROM_SERVER: /Agent_Form query_result_iterative query0 1 10 "Tenuto Silence None Staccato Tenuto"
FROM_SERVER: /Agent_Form query_result_iterative query0 2 10 "Staccato None None None Staccato None None Staccato"
(...)
FROM_SERVER: /Agent_Form query_result_iterative query0 10 10 "Staccato None None None Staccato None None Staccato"

__Note that the output of the query is sent event by event with a counter indicating the potion of the event in the generated sequence__ 

### Second agent "/Agent_Audio"

In this example we chain "/Agent_Form" and "/Agent_Audio": the "content" alphabet of one corresponding to the "label" alphabet of the other, we can concatenate the output of the query to "/Agent_Form" to use it as a "scenario" for the query to "/Agent_Audio".

#### Send a "scenario" query to the first agent "/Agent_Audio"

*The run of a "scenario" query will output the best possible match for a given sequence of labels (see related research articles).*

TO_SERVER: /Agent_Audio query query0 0 relative None listlabel [ Tenuto Silence None Staccato Tenuto Staccato None None None Staccato None None Staccato Tenuto Silence None Staccato Tenuto Staccato None None None Staccato None None Staccato Tenuto Silence None Staccato Tenuto Staccato None None None Staccato None None Staccato Tenuto Silence None Staccato Tenuto Staccato None None None Staccato None None Staccato Tenuto Silence None Staccato Tenuto Staccato None None None Staccato None None Staccato ]

__Note the syntax of a "scenario" query :
`{name_agent} query {name_query} {start_date} {relative or absolute} None {[ list_of_labels_to_match ] }`__

__The "None" keyword allows the generation to chose an optimal label.__

#### Receiving and parsing the result of the query

FROM_SERVER: /Agent_Audio server_received_query query0
FROM_SERVER: /Agent_Audio query_result_iterative query0 1 65 "52458 54795"
FROM_SERVER: /Agent_Audio query_result_iterative query0 2 65 "84360 85772"
FROM_SERVER: /Agent_Audio query_result_iterative query0 3 65 "85772 86997"
(...)
FROM_SERVER: /Agent_Audio query_result_iterative query0 64 65 "19330 19538"
FROM_SERVER: /Agent_Audio query_result_iterative query0 65 65 "19869 20169"

__Note that the output of the query is sent event by event with a counter indicating the potion of the event in the generated sequence__ 


Finally, the result of the process is a sequence of markers that can be used to drive concatenative synthesis:
- it matches a sequence of vocal playing modes in an optimal way thanks to the model learned in the 2nd agent
- and this sequence of vocal playing modes has itself been generated by the system in the style of a higher level corpus associating categories of abstract musical gestures to sequences of vocal playing modes.

## Exit

TO_SERVER: /server exit

FROM_SERVER: /Agent_Audio terminated bang
FROM_SERVER: /Agent_Form terminated bang
FROM_SERVER: /server info "DICY2 server terminated"
FROM_SERVER: /server terminated bang



