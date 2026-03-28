# Using Circumplex Instruments


Source: https://circumplex.jmgirard.com/articles/using-instruments.html

``` python
from great_tables import GT

import circumplex

%load_ext autoreload
%autoreload 2
```

## Overview of Instrument-related Functions

Although the circumplex package is capable of analyzing and visualizing
data in a “source-agnostic” manner (i.e., without knowing what the
numbers correspond to), it can be helpful to both the user and the
package to have more contextual information about which
information/questionnaire the data come from. For example, knowing the
specific instrument used can enable the package to automatically score
item-level responses and standardize these scores using normative data.
Furthermore, a centralized repository of information about circumplex
instruments would provide a convenient and accessible way for users to
discover and begin using new instruments.

The first part of this tutorial will discuss how to preview the
instruments currently available in the circumplex package, how to load
information about a specific instrument for use in analysis, and how to
extract general and specific information about that instrument. The
following functions will be discussed: `instruments()`, `instrument()`,
`print()`, `summary()`, `scales()`, `items()`, `anchors()`, `norms()`,
and `View()`.

The second part of this tutorial will discuss how to use the information
about an instrument to transform and summarize circumplex data. It will
demonstrate how to ipsatize item-level responses (i.e. apply deviation
scoring across variables), how to calculate scale scores from item-level
responses (with or without imputing/prorating missing values), and how
to standardize scale scores using normative/comparison data. The
following functions will be discussed: `ipsatize()`, `score()`, and
`standardize()`.

## 2. Loading and Examining Instrument Objects

### Previewing the available instruments

You can preview the list of currently available instruments using the
`instruments()` function. This function will print the abbreviation,
name, and (in parentheses) the “code” for each available instrument. We
will return to the code in the next section.

``` python
circumplex.show_instruments()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">          The circumplex package currently includes 3 instruments          </span>
┏━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">   </span>┃<span style="font-weight: bold"> Abbreviation </span>┃<span style="font-weight: bold"> Name                                                 </span>┃
┡━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1 │<span style="color: #008080; text-decoration-color: #008080"> CSIG         </span>│<span style="color: #800080; text-decoration-color: #800080"> Circumplex Scales of Interpersonal Goals             </span>│
│ 2 │<span style="color: #008080; text-decoration-color: #008080"> IIPSC        </span>│<span style="color: #800080; text-decoration-color: #800080"> Inventory of Interpersonal Problems Short Circumplex </span>│
│ 3 │<span style="color: #008080; text-decoration-color: #008080"> IPIPIPC      </span>│<span style="color: #800080; text-decoration-color: #800080"> IPIP Interpersonal Circumplex                        </span>│
└───┴──────────────┴──────────────────────────────────────────────────────┘
</pre>

### Loading a specific instrument

To reduce loading time and memory usage, instrument information is not
loaded into memory when the circumplex package is loaded. Instead,
instruments should be loaded into memory on an as-needed bases. As
demonstrated below, this can be done by passing an instrument’s code
(which we saw how to find in the last section) to the
`load_instrument()` function. We can then examine that instrument data
using the `print()` function.

``` python
csig = circumplex.get_instrument("csig")
print(csig)
```

    CSIG: Circumplex Scales of Interpersonal Goals
    32 items, 8 scales, 1 normative data sets
    Lock (2014)
    < https://doi.org/10.1177/0146167213514280 >

### Examining an instrument in-depth

To examine the information available about a loaded instrument, there
are several options. To print a long list of formatted information about
the instrument, use the `summary()` function. This will return the same
information returned by `print()`, followed by information about the
instrument’s scales, rating scale anchors, items, and normative data
set(s). The summary of each instrument is also available from the
package reference page.

``` python
csig.info()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">CSIG: Circumplex Scales of Interpersonal Goals
<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span> items, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span> scales, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> normative data sets
Lock <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2014</span><span style="font-weight: bold">)</span>
<span style="font-weight: bold">&lt;</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://doi.org/10.1177/0146167213514280</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="font-weight: bold">&gt;</span>


<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">The CSIG contains 8 scales:</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">PA (90°): Be authoritative</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">BC (135°): Be tough</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">DE (180°): Be self-protective</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">FG (225°): Be wary</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">HI (270°): Be conflict-avoidant</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">JK (315°): Be cooperative</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">LM (360°): Be understanding</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">└── </span><span style="font-weight: bold">NO (45°): Be respected</span>


<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">The CSIG is rated using the following 5-point scale:</span>
  0. It is not at all important that...
  1. It is somewhat important that...
  2. It is moderately important that...
  3. It is very important that...
  4. It is extremely important that...



<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">The CSIG currently has 1 normative data set(s):</span>

1. 665 MTurkers from US, Canada, and India about interactions between nations
   Lock (2014)
   https://doi.org/10.1177/0146167213514280

</pre>

Specific subsections of this output can be returned individually by
printing the `scales`, `anchors`, `items`, and `norms` attributes of the
instrument object. These functions are especially useful when you need
to know a specific bit of information about an instrument and don’t want
the console to be flooded with unneeded information.

``` python
csig.info_anchors()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">The CSIG is rated using the following 5-point scale:</span>
  0. It is not at all important that...
  1. It is somewhat important that...
  2. It is moderately important that...
  3. It is very important that...
  4. It is extremely important that...

</pre>

Some of these attributes also have additional methods to customize their
output. For instance, the `scales` attribute has a `.show()` method
which includes the option to display the items for each scale.

``` python
csig.info_scales(items=True)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">The CSIG contains 8 scales:</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">PA (90°): Be authoritative</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">8. We are assertive</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">16. We appear confident</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">24. We are decisive</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">32. They see us as capable</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">BC (135°): Be tough</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">5. We show that we can be tough</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">12. They not get angry with us</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">21. We are aggressive if necessary</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">29. We not show our weaknesses</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">DE (180°): Be self-protective</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">2. We are the winners in any argument or dispute</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">10. We do whatever is in our best interest</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">18. We are better than them</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">26. We keep our guard up</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">FG (225°): Be wary</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">7. We let them fend for themselves</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">15. They stay out of our business</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">23. We not trust them</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">31. We not get entangled in their affairs</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">HI (270°): Be conflict-avoidant</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">4. We avoid conflict</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">12. They not get angry with us</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">20. We not get into arguments</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">28. We not make them angry</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">JK (315°): Be cooperative</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">1. We are friendly</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">9. We celebrate their achievements</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">17. They feel we are all on the same team</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">25. We are cooperative</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">LM (360°): Be understanding</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">6. We appreciate what they have to offer</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">14. We understand their point of view</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">22. We show concern for their welfare</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">30. We are able to compromise</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">└── </span><span style="font-weight: bold">NO (45°): Be respected</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">    ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">3. They respect what we have to say</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">    ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">11. We get the chance to express our views</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">    ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">19. They listen to what we have to say</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">    └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">27. They see us as responsible</span>
</pre>

## 3. Instrument-related tidying functions

It is a good idea in practice to digitize and save each participant’s
response to each item on an instrument, rather than just their scores on
each scale. Having access to item-level data will make it easier to spot
and correct mistakes, will enable more advanced analysis of missing
data, and will enable latent variable models that account for
measurement error (e.g. structural equation modelling). Furthermore, the
functions described below will make it easy to transform and summarize
such item-level data into scale scores.

First, however, we need to make sure the item-level data is in the
expected format. Your data should be stored in a data frame where each
row corresponds to one observation (e.g. participant, organization, or
timepoint) and each column corresponds to one variable describing these
observations (e.g. item responses, demographic characteristics, scale
scores). The `pandas` package provides excellent tools for getting your
data into this format from a variety of different file types and
formats.

For the purpose of illustration, we will work with a small-scale data
set, which includes item-level responses to the Inventory of
Interpersonal Problems, Short Circumplex (IIP-SC) for just 10
participants. As will become important later on, this data set contains
a small amount of missing values (represented as `NA`). This data set is
included as part of the circumplex package and can be loaded and
previewed as follows:

``` python
from circumplex import load_dataset

raw_jz2017 = load_dataset("jz2017")
GT(raw_jz2017.head())
```

<div id="eogabipoex" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#eogabipoex table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#eogabipoex thead, tbody, tfoot, tr, td, th { border-style: none !important; }
 tr { background-color: transparent !important; }
#eogabipoex p { margin: 0 !important; padding: 0 !important; }
 #eogabipoex .gt_table { display: table !important; border-collapse: collapse !important; line-height: normal !important; margin-left: auto !important; margin-right: auto !important; color: #333333 !important; font-size: 16px !important; font-weight: normal !important; font-style: normal !important; background-color: #FFFFFF !important; width: auto !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #A8A8A8 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #A8A8A8 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; }
 #eogabipoex .gt_caption { padding-top: 4px !important; padding-bottom: 4px !important; }
 #eogabipoex .gt_title { color: #333333 !important; font-size: 125% !important; font-weight: initial !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; border-bottom-color: #FFFFFF !important; border-bottom-width: 0 !important; }
 #eogabipoex .gt_subtitle { color: #333333 !important; font-size: 85% !important; font-weight: initial !important; padding-top: 3px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; border-top-color: #FFFFFF !important; border-top-width: 0 !important; }
 #eogabipoex .gt_heading { background-color: #FFFFFF !important; text-align: center !important; border-bottom-color: #FFFFFF !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #eogabipoex .gt_bottom_border { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #eogabipoex .gt_col_headings { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #eogabipoex .gt_col_heading { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; overflow-x: hidden !important; }
 #eogabipoex .gt_column_spanner_outer { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; padding-top: 0 !important; padding-bottom: 0 !important; padding-left: 4px !important; padding-right: 4px !important; }
 #eogabipoex .gt_column_spanner_outer:first-child { padding-left: 0 !important; }
 #eogabipoex .gt_column_spanner_outer:last-child { padding-right: 0 !important; }
 #eogabipoex .gt_column_spanner { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; overflow-x: hidden !important; display: inline-block !important; width: 100% !important; }
 #eogabipoex .gt_spanner_row { border-bottom-style: hidden !important; }
 #eogabipoex .gt_group_heading { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; text-align: left !important; }
 #eogabipoex .gt_empty_group_heading { padding: 0.5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: middle !important; }
 #eogabipoex .gt_from_md> :first-child { margin-top: 0 !important; }
 #eogabipoex .gt_from_md> :last-child { margin-bottom: 0 !important; }
 #eogabipoex .gt_row { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; margin: 10px !important; border-top-style: solid !important; border-top-width: 1px !important; border-top-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; overflow-x: hidden !important; }
 #eogabipoex .gt_stub { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; }
 #eogabipoex .gt_stub_row_group { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; vertical-align: top !important; }
 #eogabipoex .gt_row_group_first td { border-top-width: 2px !important; }
 #eogabipoex .gt_row_group_first th { border-top-width: 2px !important; }
 #eogabipoex .gt_striped { color: #333333 !important; background-color: #F4F4F4 !important; }
 #eogabipoex .gt_table_body { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #eogabipoex .gt_grand_summary_row { color: #333333 !important; background-color: #FFFFFF !important; text-transform: inherit !important; padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; }
 #eogabipoex .gt_first_grand_summary_row_bottom { border-top-style: double !important; border-top-width: 6px !important; border-top-color: #D3D3D3 !important; }
 #eogabipoex .gt_last_grand_summary_row_top { border-bottom-style: double !important; border-bottom-width: 6px !important; border-bottom-color: #D3D3D3 !important; }
 #eogabipoex .gt_sourcenotes { color: #333333 !important; background-color: #FFFFFF !important; border-bottom-style: none !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; }
 #eogabipoex .gt_sourcenote { font-size: 90% !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; text-align: left !important; }
 #eogabipoex .gt_left { text-align: left !important; }
 #eogabipoex .gt_center { text-align: center !important; }
 #eogabipoex .gt_right { text-align: right !important; font-variant-numeric: tabular-nums !important; }
 #eogabipoex .gt_font_normal { font-weight: normal !important; }
 #eogabipoex .gt_font_bold { font-weight: bold !important; }
 #eogabipoex .gt_font_italic { font-style: italic !important; }
 #eogabipoex .gt_super { font-size: 65% !important; }
 #eogabipoex .gt_footnote_marks { font-size: 75% !important; vertical-align: 0.4em !important; position: initial !important; }
 #eogabipoex .gt_asterisk { font-size: 100% !important; vertical-align: 0 !important; }

</style>

<table class="gt_table" data-quarto-postprocess="true"
data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="Gender" class="gt_col_heading gt_columns_bottom_border gt_left"
data-quarto-table-cell-role="th" scope="col">Gender</th>
<th id="PA" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">PA</th>
<th id="BC" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">BC</th>
<th id="DE" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">DE</th>
<th id="FG" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">FG</th>
<th id="HI" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">HI</th>
<th id="JK" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">JK</th>
<th id="LM" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">LM</th>
<th id="NO" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">NO</th>
<th id="PARPD" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">PARPD</th>
<th id="SCZPD" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">SCZPD</th>
<th id="SZTPD" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">SZTPD</th>
<th id="ASPD" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">ASPD</th>
<th id="BORPD" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">BORPD</th>
<th id="HISPD" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">HISPD</th>
<th id="NARPD" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">NARPD</th>
<th id="AVPD" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">AVPD</th>
<th id="DPNPD" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">DPNPD</th>
<th id="OCPD" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">OCPD</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left">Female</td>
<td class="gt_row gt_right">1.5</td>
<td class="gt_row gt_right">1.5</td>
<td class="gt_row gt_right">1.25</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">2.5</td>
<td class="gt_row gt_right">2.25</td>
<td class="gt_row gt_right">2.5</td>
<td class="gt_row gt_right">4</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">7</td>
<td class="gt_row gt_right">7</td>
<td class="gt_row gt_right">8</td>
<td class="gt_row gt_right">4</td>
<td class="gt_row gt_right">6</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">4</td>
<td class="gt_row gt_right">6</td>
</tr>
<tr>
<td class="gt_row gt_left">Female</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.25</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.25</td>
<td class="gt_row gt_right">1.25</td>
<td class="gt_row gt_right">1.75</td>
<td class="gt_row gt_right">2.25</td>
<td class="gt_row gt_right">2.25</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
</tr>
<tr>
<td class="gt_row gt_left">Female</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">4</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">5</td>
<td class="gt_row gt_right">4</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
</tr>
<tr>
<td class="gt_row gt_left">Male</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">1.75</td>
<td class="gt_row gt_right">1.75</td>
<td class="gt_row gt_right">2.5</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">1.75</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">2.5</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
</tr>
<tr>
<td class="gt_row gt_left">Female</td>
<td class="gt_row gt_right">0.25</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.25</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
</tr>
</tbody>
</table>

</div>

``` python
iipsc = circumplex.get_instrument("iipsc")
iipsc.info(items=True)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">IIP-SC: Inventory of Interpersonal Problems Short Circumplex
<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span> items, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span> scales, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span> normative data sets
Soldz, Budman, Demby, &amp; Merry <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1995</span><span style="font-weight: bold">)</span>
<span style="font-weight: bold">&lt;</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://doi.org/10.1177/1073191195002001006</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="font-weight: bold">&gt;</span>


<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">The IIP-SC contains 8 scales:</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">PA (90°): Domineering</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">1. ...point of view...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">9. ...too aggressive toward...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">17. ...control other people...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">25. ...argue with other...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">BC (135°): Vindictive</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">2. ...supportive of another...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">10. ...another person's happiness...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">18. ...too suspicious of...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">26. ...revenge against people...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">DE (180°): Cold</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">3. ...show affection to...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">11. ...feeling of love...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">19. ...feel close to...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">27. ...at a distance...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">FG (225°): Socially avoidant</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">4. ...join in on...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">12. ...introduce myself to...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">20. ...socialize with other...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">28. ...get together socially...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">HI (270°): Nonassertive</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">5. ...stop bothering me...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">13. ...confront people with...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">21. ...assertive with another...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">29. ...to be firm...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">JK (315°): Exploitable</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">6. ...I am angry...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">14. ...assertive without worrying...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">22. ...too easily persuaded...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">30. ...people take advantage...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">LM (360°): Overly nurturant</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">7. ...my own welfare...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">15. ...please other people...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">23. ...other people's needs...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">31. ...another person's misery...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">└── </span><span style="font-weight: bold">NO (45°): Intrusive</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">    ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">8. ...keep things private...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">    ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">16. ...open up to...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">    ├── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">24. ...noticed too much...</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">    └── </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">32. ...tell personal things...</span>


<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">The IIP-SC is rated using the following 5-point scale:</span>
  0. Not at all
  1. Somewhat
  2. Moderately
  3. Very
  4. Extremely



<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">The IIP-SC currently has 2 normative data set(s):</span>

1. 872 American college students
   Hopwood, Pincus, DeMoor, &amp; Koonce (2011)
   https://doi.org/10.1080/00223890802388665
2. 106 American psychiatric outpatients
   Soldz, Budman, Demby, &amp; Merry (1995)
   https://doi.org/10.1177/1073191195002001006

</pre>
