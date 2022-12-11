# Example of OSC messages to / from Dicy2 server 

This example comes from `tab 2` in tutorial `2_Chaining_Agents.maxpat` in [Dicy2 for Max](https://forum.ircam.fr/projects/detail/dicy2/): In this patch we chain two agents trained on data representing different levels of musical hierarchy to create a composition process that operates at multiple scales. 

Agent_Audio is trained on the labeled segments we have previously encountered in our first audio tutorial, and will supply the final output. Agent_Form is trained on a corpus of musical sequences comprised of the same labels, thus creating a set of higher-level gestures. 

Scenarios sent to Agent_Form will generate sequences of these gestures, which are in turn passed to Agent_Audio as scenarios. This hierarchical model can be extended to any numer of levels, and the general approach can be applied to any use case such as MIDI or other modes of audio labeling.


---

## Launch server

Launch `dicy2_server.app` with OSC ports as arguments
*NB: this applies of course to `dicy2_server.app` or to a server started from the Python code!*

__FROM_SERVER:__

/server initialized

__FROM_SERVER:__

/server status bang *... (then looped)*

---

## 1 - Create two agents


### 1.1 - Create first agent (labels=abstract musical gesture types, contents=sequences of vocal playing modes)

#### 1.1.1 - Create first agent "Agent_Form"

__TO_SERVER:__

/server create_agent /Agent_Form listlabel

FROM_SERVER:

/server info "Created new agent at '/Agent_Form'"

__FROM_SERVER:__

/server create_agent /Agent_Form

__FROM_SERVER:__

/Agent_Form initialized

__FROM_SERVER:__

/Agent_Form status bang 
*... (then looped)*
 

#### 1.1.2 - Setting some parameters for first agent (in particular the max_continuity parameters)

__TO_SERVER:__

/Agent_Form set_control_parameter generator::prospector::\_navigator::max_continuity 6

__TO_SERVER:__

/Agent_Form set_control_parameter generator::force_output 1

__TO_SERVER:__

/Agent_Form clear listlabel

__FROM_SERVER:__

/Agent_Form clear ListLabel ... 
(+ list of default paramaters)


#### 1.1.3 - Learning a sequence in first agent (labels=abstract musical gesture types, contents=sequences of vocal playing modes)

__TO_SERVER:__

/Agent_Form learn_event listlabel Gesture_type_1 "Agilita Breath Ribatutto Silence Staccato Tenuto"

__TO_SERVER:__

/Agent_Form learn_event listlabel Gesture_type_2 "Staccato Staccato Staccato"

(... send other couples {label} {content})

__FROM_SERVER:__

/Agent_Form new_event_learned "Agilita Breath Ribatutto Silence Staccato Tenuto"

__FROM_SERVER:__

/Agent_Form info "Learned event 'Agilita Breath Ribatutto Silence Staccato Tenuto'"

__FROM_SERVER:__

/Agent_Form new_event_learned "Staccato Staccato Staccato"

__FROM_SERVER:__

/Agent_Form info "Learned event 'Staccato Staccato Staccato'"
(... feedback other couples {label} {content})

---

### 1.2 - Create second agent (labels=vocal playing mode, contents=corresponding markers in an audio file)

#### 1.2.1 - Create second agent "Agent_Audio"

__TO_SERVER:__

/server create_agent /Agent_Audio listlabel

FROM_SERVER:

/server info "Created new agent at '/Agent_Audio'"

__FROM_SERVER:__

/server create_agent /Agent_Audio

__FROM_SERVER:__

/Agent_Audio initialized

__FROM_SERVER:__

/Agent_Audio status bang 
*... (then looped)*

####  1.2.2 - Setting some parameters for second agent (in particular the max_continuity parameters)

__TO_SERVER:__

/Agent_Audio set_control_parameter generator::prospector::\_navigator::max_continuity 6

__TO_SERVER:__

/Agent_Audio set_control_parameter generator::force_output 1

__FROM_SERVER:__

/Agent_Audio clear ListLabel ... (+ list of default paramaters)

####  1.2.3 - Learning a sequence in first agent (labels=vocal playing mode, contents=corresponding markers in an audio file)

__TO_SERVER:__

/Agent_Audio clear listlabel

__TO_SERVER:__

/Agent_Audio learn_event listlabel Tenuto "0 1876"

__TO_SERVER:__

/Agent_Audio learn_event listlabel Agilita "1876 2398"

(... send other couples {label} {content})

__FROM_SERVER:__

/Agent_Audio new_event_learned "0 1876"

__FROM_SERVER:__

/Agent_Audio info "Learned event '0 1876'"

__FROM_SERVER:__

/Agent_Audio new_event_learned "1876 2398"

__FROM_SERVER:__

/Agent_Audio info "Learned event '1876 2398'"

(... feedback other couples {label} {content})

---

##  2 - Send queries to the agents


### 2.1 - First agent "/Agent_Form"

#### 2.1.1 - Send a "free" query to the first agent "/Agent_Form"

*A "free" query takes advantage of the model built on the sequence to generate a new sequence without any constraint than the `maxcontinuity` value defined earlier*

__TO_SERVER:__/Agent_Form query query0 0 relative 10 

__Note the syntax of a "free" query :
`{name_agent} query {name_query} {start_date} {relative or absolute} {number_of_events_to_generate}`__

#### 2.1.2 - Receiving and parsing the result of the query

__FROM_SERVER:__

/Agent_Form server_received_query query0

__FROM_SERVER:__

/Agent_Form query_result_iterative query0 1 10 "Tenuto Silence None Staccato Tenuto"

FROM_SERVER:

/Agent_Form query_result_iterative query0 2 10 "Staccato None None None Staccato None None Staccato"

(...)

__FROM_SERVER:__

/Agent_Form query_result_iterative query0 10 10 "Staccato None None None Staccato None None Staccato"

__Note that the output of the query is sent event by event with a counter indicating the potion of the event in the generated sequence__ 

---

### 2.2 - Second agent "/Agent_Audio"

In this example we chain "/Agent_Form" and "/Agent_Audio": the "content" alphabet of one corresponding to the "label" alphabet of the other, we can concatenate the output of the query to "/Agent_Form" to use it as a "scenario" for the query to "/Agent_Audio".

#### 2.2.1 - Send a "scenario" query to the first agent "/Agent_Audio"

*The run of a "scenario" query will output the best possible match for a given sequence of labels (see related research articles).*

__TO_SERVER:__/Agent_Audio query query0 0 relative None listlabel \[ Tenuto Silence None Staccato Tenuto Staccato None None None Staccato None None Staccato Tenuto Silence None Staccato Tenuto Staccato None None None Staccato None None Staccato Tenuto Silence None Staccato Tenuto Staccato None None None Staccato None None Staccato Tenuto Silence None Staccato Tenuto Staccato None None None Staccato None None Staccato Tenuto Silence None Staccato Tenuto Staccato None None None Staccato None None Staccato \]

__Note the syntax of a "scenario" query :
`{name_agent} query {name_query} {start_date} {relative or absolute} None {[ list_of_labels_to_match ] }`__

__The "None" keyword allows the generation to chose an optimal label.__

#### 2.2.2 - Receiving and parsing the result of the query

__FROM_SERVER:__

/Agent_Audio server_received_query query0

__FROM_SERVER:__

/Agent_Audio query_result_iterative query0 1 65 "52458 54795"

__FROM_SERVER:__

/Agent_Audio query_result_iterative query0 2 65 "84360 85772"

__FROM_SERVER:__

/Agent_Audio query_result_iterative query0 3 65 "85772 86997"

(...)

__FROM_SERVER:__

/Agent_Audio query_result_iterative query0 64 65 "19330 19538"

__FROM_SERVER:__

/Agent_Audio query_result_iterative query0 65 65 "19869 20169"

__Note that the output of the query is sent event by event with a counter indicating the potion of the event in the generated sequence__ 

#### 2.2.3 - Final output

Finally, the result of the process is a sequence of markers that can be used to drive concatenative synthesis:
- it matches a sequence of vocal playing modes in an optimal way thanks to the model learned in the 2nd agent
- and this sequence of vocal playing modes has itself been generated by the system in the style of a higher level corpus associating categories of abstract musical gestures to sequences of vocal playing modes.

---

## 3 - Exit

__TO_SERVER:__

/server exit

__FROM_SERVER:__

/Agent_Audio terminated bang

FROM_SERVER:

/Agent_Form terminated bang

__FROM_SERVER:__

/server info "DICY2 server terminated"

__FROM_SERVER:__

/server terminated bang



