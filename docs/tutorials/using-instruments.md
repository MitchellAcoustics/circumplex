# Using Circumplex Instruments


``` python
%load_ext rich
import numpy as np
from great_tables import GT

from circumplex import (
    get_instrument,
    ipsatize,
    norm_standardize,
    score,
    show_instruments,
)
```

    The rich extension is already loaded. To reload it, use:
      %reload_ext rich

## 1. Overview of Instrument-related Functions

## 2. Loading and Examining Instrument Objects

### Previewing the available instruments

``` python
show_instruments()
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

``` python
csig = get_instrument("csig")
csig
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>


    Instrument(
        'CSIG: Circumplex Scales of Interpersonal Goals',
        '32 items, 8 scales, 1 normative data sets',
        'Lock (2014)',
        '< https://doi.org/10.1177/0146167213514280 >'
    )

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

## 3. Instrument-related Tidying Functions

``` python
iipsc = get_instrument("iipsc")
iipsc.info()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">IIP-SC: Inventory of Interpersonal Problems Short Circumplex
<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span> items, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span> scales, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span> normative data sets
Soldz, Budman, Demby, &amp; Merry <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1995</span><span style="font-weight: bold">)</span>
<span style="font-weight: bold">&lt;</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://doi.org/10.1177/1073191195002001006</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="font-weight: bold">&gt;</span>


<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">The IIP-SC contains 8 scales:</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">PA (90°): Domineering</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">BC (135°): Vindictive</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">DE (180°): Cold</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">FG (225°): Socially avoidant</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">HI (270°): Nonassertive</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">JK (315°): Exploitable</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">LM (360°): Overly nurturant</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">└── </span><span style="font-weight: bold">NO (45°): Intrusive</span>


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

``` python
import pandas as pd

raw_iipsc = pd.read_csv(
    "/Users/mitch/Documents/GitHub/python-circumplex/src/circumplex/data/raw_iipsc.csv"
)
GT(raw_iipsc.head())
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
<div id="dcyxntnosi" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#dcyxntnosi table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#dcyxntnosi thead, tbody, tfoot, tr, td, th { border-style: none !important; }
 tr { background-color: transparent !important; }
#dcyxntnosi p { margin: 0 !important; padding: 0 !important; }
 #dcyxntnosi .gt_table { display: table !important; border-collapse: collapse !important; line-height: normal !important; margin-left: auto !important; margin-right: auto !important; color: #333333 !important; font-size: 16px !important; font-weight: normal !important; font-style: normal !important; background-color: #FFFFFF !important; width: auto !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #A8A8A8 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #A8A8A8 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; }
 #dcyxntnosi .gt_caption { padding-top: 4px !important; padding-bottom: 4px !important; }
 #dcyxntnosi .gt_title { color: #333333 !important; font-size: 125% !important; font-weight: initial !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; border-bottom-color: #FFFFFF !important; border-bottom-width: 0 !important; }
 #dcyxntnosi .gt_subtitle { color: #333333 !important; font-size: 85% !important; font-weight: initial !important; padding-top: 3px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; border-top-color: #FFFFFF !important; border-top-width: 0 !important; }
 #dcyxntnosi .gt_heading { background-color: #FFFFFF !important; text-align: center !important; border-bottom-color: #FFFFFF !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #dcyxntnosi .gt_bottom_border { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #dcyxntnosi .gt_col_headings { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #dcyxntnosi .gt_col_heading { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; overflow-x: hidden !important; }
 #dcyxntnosi .gt_column_spanner_outer { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; padding-top: 0 !important; padding-bottom: 0 !important; padding-left: 4px !important; padding-right: 4px !important; }
 #dcyxntnosi .gt_column_spanner_outer:first-child { padding-left: 0 !important; }
 #dcyxntnosi .gt_column_spanner_outer:last-child { padding-right: 0 !important; }
 #dcyxntnosi .gt_column_spanner { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; overflow-x: hidden !important; display: inline-block !important; width: 100% !important; }
 #dcyxntnosi .gt_spanner_row { border-bottom-style: hidden !important; }
 #dcyxntnosi .gt_group_heading { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; text-align: left !important; }
 #dcyxntnosi .gt_empty_group_heading { padding: 0.5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: middle !important; }
 #dcyxntnosi .gt_from_md> :first-child { margin-top: 0 !important; }
 #dcyxntnosi .gt_from_md> :last-child { margin-bottom: 0 !important; }
 #dcyxntnosi .gt_row { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; margin: 10px !important; border-top-style: solid !important; border-top-width: 1px !important; border-top-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; overflow-x: hidden !important; }
 #dcyxntnosi .gt_stub { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; }
 #dcyxntnosi .gt_stub_row_group { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; vertical-align: top !important; }
 #dcyxntnosi .gt_row_group_first td { border-top-width: 2px !important; }
 #dcyxntnosi .gt_row_group_first th { border-top-width: 2px !important; }
 #dcyxntnosi .gt_striped { color: #333333 !important; background-color: #F4F4F4 !important; }
 #dcyxntnosi .gt_table_body { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #dcyxntnosi .gt_grand_summary_row { color: #333333 !important; background-color: #FFFFFF !important; text-transform: inherit !important; padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; }
 #dcyxntnosi .gt_first_grand_summary_row_bottom { border-top-style: double !important; border-top-width: 6px !important; border-top-color: #D3D3D3 !important; }
 #dcyxntnosi .gt_last_grand_summary_row_top { border-bottom-style: double !important; border-bottom-width: 6px !important; border-bottom-color: #D3D3D3 !important; }
 #dcyxntnosi .gt_sourcenotes { color: #333333 !important; background-color: #FFFFFF !important; border-bottom-style: none !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; }
 #dcyxntnosi .gt_sourcenote { font-size: 90% !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; text-align: left !important; }
 #dcyxntnosi .gt_left { text-align: left !important; }
 #dcyxntnosi .gt_center { text-align: center !important; }
 #dcyxntnosi .gt_right { text-align: right !important; font-variant-numeric: tabular-nums !important; }
 #dcyxntnosi .gt_font_normal { font-weight: normal !important; }
 #dcyxntnosi .gt_font_bold { font-weight: bold !important; }
 #dcyxntnosi .gt_font_italic { font-style: italic !important; }
 #dcyxntnosi .gt_super { font-size: 65% !important; }
 #dcyxntnosi .gt_footnote_marks { font-size: 75% !important; vertical-align: 0.4em !important; position: initial !important; }
 #dcyxntnosi .gt_asterisk { font-size: 100% !important; vertical-align: 0 !important; }

</style>

<table class="gt_table" data-quarto-postprocess="true"
data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="IIP01" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP01</th>
<th id="IIP02" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP02</th>
<th id="IIP03" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP03</th>
<th id="IIP04" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP04</th>
<th id="IIP05" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP05</th>
<th id="IIP06" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP06</th>
<th id="IIP07" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP07</th>
<th id="IIP08" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP08</th>
<th id="IIP09" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP09</th>
<th id="IIP10" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP10</th>
<th id="IIP11" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP11</th>
<th id="IIP12" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP12</th>
<th id="IIP13" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP13</th>
<th id="IIP14" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP14</th>
<th id="IIP15" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP15</th>
<th id="IIP16" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP16</th>
<th id="IIP17" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP17</th>
<th id="IIP18" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP18</th>
<th id="IIP19" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP19</th>
<th id="IIP20" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP20</th>
<th id="IIP21" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP21</th>
<th id="IIP22" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP22</th>
<th id="IIP23" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP23</th>
<th id="IIP24" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP24</th>
<th id="IIP25" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP25</th>
<th id="IIP26" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP26</th>
<th id="IIP27" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP27</th>
<th id="IIP28" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP28</th>
<th id="IIP29" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP29</th>
<th id="IIP30" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP30</th>
<th id="IIP31" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP31</th>
<th id="IIP32" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">IIP32</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">4</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">4</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
</tr>
<tr>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">4</td>
<td class="gt_row gt_right">3.0</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">2</td>
</tr>
<tr>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3.0</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
</tr>
<tr>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right"><na></td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">4</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">2</td>
</tr>
<tr>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">3</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">2</td>
<td class="gt_row gt_right">1</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
<td class="gt_row gt_right">0</td>
</tr>
</tbody>
</table>

</div>

### Ipsatizing item-level data

``` python
ips_iipsc = ipsatize(
    data=raw_iipsc,
    items=np.arange(0, 32),
    append=False,
)
GT(ips_iipsc.round(2))
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
<div id="abzxxgevsq" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#abzxxgevsq table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#abzxxgevsq thead, tbody, tfoot, tr, td, th { border-style: none !important; }
 tr { background-color: transparent !important; }
#abzxxgevsq p { margin: 0 !important; padding: 0 !important; }
 #abzxxgevsq .gt_table { display: table !important; border-collapse: collapse !important; line-height: normal !important; margin-left: auto !important; margin-right: auto !important; color: #333333 !important; font-size: 16px !important; font-weight: normal !important; font-style: normal !important; background-color: #FFFFFF !important; width: auto !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #A8A8A8 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #A8A8A8 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; }
 #abzxxgevsq .gt_caption { padding-top: 4px !important; padding-bottom: 4px !important; }
 #abzxxgevsq .gt_title { color: #333333 !important; font-size: 125% !important; font-weight: initial !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; border-bottom-color: #FFFFFF !important; border-bottom-width: 0 !important; }
 #abzxxgevsq .gt_subtitle { color: #333333 !important; font-size: 85% !important; font-weight: initial !important; padding-top: 3px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; border-top-color: #FFFFFF !important; border-top-width: 0 !important; }
 #abzxxgevsq .gt_heading { background-color: #FFFFFF !important; text-align: center !important; border-bottom-color: #FFFFFF !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #abzxxgevsq .gt_bottom_border { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #abzxxgevsq .gt_col_headings { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #abzxxgevsq .gt_col_heading { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; overflow-x: hidden !important; }
 #abzxxgevsq .gt_column_spanner_outer { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; padding-top: 0 !important; padding-bottom: 0 !important; padding-left: 4px !important; padding-right: 4px !important; }
 #abzxxgevsq .gt_column_spanner_outer:first-child { padding-left: 0 !important; }
 #abzxxgevsq .gt_column_spanner_outer:last-child { padding-right: 0 !important; }
 #abzxxgevsq .gt_column_spanner { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; overflow-x: hidden !important; display: inline-block !important; width: 100% !important; }
 #abzxxgevsq .gt_spanner_row { border-bottom-style: hidden !important; }
 #abzxxgevsq .gt_group_heading { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; text-align: left !important; }
 #abzxxgevsq .gt_empty_group_heading { padding: 0.5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: middle !important; }
 #abzxxgevsq .gt_from_md> :first-child { margin-top: 0 !important; }
 #abzxxgevsq .gt_from_md> :last-child { margin-bottom: 0 !important; }
 #abzxxgevsq .gt_row { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; margin: 10px !important; border-top-style: solid !important; border-top-width: 1px !important; border-top-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; overflow-x: hidden !important; }
 #abzxxgevsq .gt_stub { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; }
 #abzxxgevsq .gt_stub_row_group { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; vertical-align: top !important; }
 #abzxxgevsq .gt_row_group_first td { border-top-width: 2px !important; }
 #abzxxgevsq .gt_row_group_first th { border-top-width: 2px !important; }
 #abzxxgevsq .gt_striped { color: #333333 !important; background-color: #F4F4F4 !important; }
 #abzxxgevsq .gt_table_body { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #abzxxgevsq .gt_grand_summary_row { color: #333333 !important; background-color: #FFFFFF !important; text-transform: inherit !important; padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; }
 #abzxxgevsq .gt_first_grand_summary_row_bottom { border-top-style: double !important; border-top-width: 6px !important; border-top-color: #D3D3D3 !important; }
 #abzxxgevsq .gt_last_grand_summary_row_top { border-bottom-style: double !important; border-bottom-width: 6px !important; border-bottom-color: #D3D3D3 !important; }
 #abzxxgevsq .gt_sourcenotes { color: #333333 !important; background-color: #FFFFFF !important; border-bottom-style: none !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; }
 #abzxxgevsq .gt_sourcenote { font-size: 90% !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; text-align: left !important; }
 #abzxxgevsq .gt_left { text-align: left !important; }
 #abzxxgevsq .gt_center { text-align: center !important; }
 #abzxxgevsq .gt_right { text-align: right !important; font-variant-numeric: tabular-nums !important; }
 #abzxxgevsq .gt_font_normal { font-weight: normal !important; }
 #abzxxgevsq .gt_font_bold { font-weight: bold !important; }
 #abzxxgevsq .gt_font_italic { font-style: italic !important; }
 #abzxxgevsq .gt_super { font-size: 65% !important; }
 #abzxxgevsq .gt_footnote_marks { font-size: 75% !important; vertical-align: 0.4em !important; position: initial !important; }
 #abzxxgevsq .gt_asterisk { font-size: 100% !important; vertical-align: 0 !important; }

</style>

<table class="gt_table" data-quarto-postprocess="true"
data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="0_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">0_i</th>
<th id="1_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">1_i</th>
<th id="2_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">2_i</th>
<th id="3_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">3_i</th>
<th id="4_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">4_i</th>
<th id="5_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">5_i</th>
<th id="6_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">6_i</th>
<th id="7_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">7_i</th>
<th id="8_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">8_i</th>
<th id="9_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">9_i</th>
<th id="10_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">10_i</th>
<th id="11_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">11_i</th>
<th id="12_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">12_i</th>
<th id="13_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">13_i</th>
<th id="14_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">14_i</th>
<th id="15_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">15_i</th>
<th id="16_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">16_i</th>
<th id="17_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">17_i</th>
<th id="18_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">18_i</th>
<th id="19_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">19_i</th>
<th id="20_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">20_i</th>
<th id="21_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">21_i</th>
<th id="22_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">22_i</th>
<th id="23_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">23_i</th>
<th id="24_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">24_i</th>
<th id="25_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">25_i</th>
<th id="26_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">26_i</th>
<th id="27_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">27_i</th>
<th id="28_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">28_i</th>
<th id="29_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">29_i</th>
<th id="30_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">30_i</th>
<th id="31_i" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">31_i</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">3.0</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">3.0</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">-1.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">-1.0</td>
</tr>
<tr>
<td class="gt_row gt_right">0.06</td>
<td class="gt_row gt_right">0.06</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">2.06</td>
<td class="gt_row gt_right">1.06</td>
<td class="gt_row gt_right">1.06</td>
<td class="gt_row gt_right">0.06</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">0.06</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">0.06</td>
<td class="gt_row gt_right">3.06</td>
<td class="gt_row gt_right">2.06</td>
<td class="gt_row gt_right">2.06</td>
<td class="gt_row gt_right">0.06</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">0.06</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">0.06</td>
<td class="gt_row gt_right">1.06</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">0.06</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">-0.94</td>
<td class="gt_row gt_right">1.06</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">-1.41</td>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">-1.41</td>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">1.59</td>
<td class="gt_row gt_right">-1.41</td>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">-1.41</td>
<td class="gt_row gt_right">-1.41</td>
<td class="gt_row gt_right">0.59</td>
<td class="gt_row gt_right">1.59</td>
<td class="gt_row gt_right">1.59</td>
<td class="gt_row gt_right">0.59</td>
<td class="gt_row gt_right">0.59</td>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">-1.41</td>
<td class="gt_row gt_right">1.59</td>
<td class="gt_row gt_right">0.59</td>
<td class="gt_row gt_right">1.59</td>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">-0.41</td>
<td class="gt_row gt_right">-1.41</td>
<td class="gt_row gt_right">1.59</td>
<td class="gt_row gt_right">0.59</td>
<td class="gt_row gt_right">1.59</td>
<td class="gt_row gt_right">0.59</td>
</tr>
<tr>
<td class="gt_row gt_right">0.71</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">0.71</td>
<td class="gt_row gt_right"><na></td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">0.71</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">0.71</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">0.71</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">1.71</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">-1.29</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">0.71</td>
<td class="gt_row gt_right">-1.29</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">-1.29</td>
<td class="gt_row gt_right">0.71</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">0.71</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">-1.29</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">0.71</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">0.71</td>
<td class="gt_row gt_right">-0.29</td>
<td class="gt_row gt_right">0.71</td>
<td class="gt_row gt_right">-0.29</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">1.38</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">2.38</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">1.38</td>
<td class="gt_row gt_right">0.38</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">-0.62</td>
<td class="gt_row gt_right">-0.62</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">0.72</td>
<td class="gt_row gt_right">0.72</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">1.72</td>
<td class="gt_row gt_right">0.72</td>
<td class="gt_row gt_right">0.72</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">0.72</td>
<td class="gt_row gt_right">0.72</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">-0.28</td>
<td class="gt_row gt_right">0.72</td>
</tr>
<tr>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">1.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">-0.5</td>
<td class="gt_row gt_right">-0.5</td>
</tr>
<tr>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">-0.74</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">-0.74</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">1.26</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">-0.74</td>
<td class="gt_row gt_right">-0.74</td>
<td class="gt_row gt_right">-0.74</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right"><na></td>
<td class="gt_row gt_right">1.26</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">-0.74</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">-0.74</td>
<td class="gt_row gt_right">-0.74</td>
<td class="gt_row gt_right">-0.74</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">-0.74</td>
<td class="gt_row gt_right">-0.74</td>
<td class="gt_row gt_right">0.26</td>
<td class="gt_row gt_right">1.26</td>
<td class="gt_row gt_right">0.26</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">1.03</td>
<td class="gt_row gt_right">1.03</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">0.03</td>
<td class="gt_row gt_right">2.03</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">0.03</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">0.03</td>
<td class="gt_row gt_right">0.03</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">1.03</td>
<td class="gt_row gt_right">1.03</td>
<td class="gt_row gt_right">1.03</td>
<td class="gt_row gt_right">0.03</td>
<td class="gt_row gt_right">1.03</td>
<td class="gt_row gt_right">1.03</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">2.03</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">0.03</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">0.03</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">-0.97</td>
<td class="gt_row gt_right">0.03</td>
<td class="gt_row gt_right">2.03</td>
<td class="gt_row gt_right">-0.97</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">1.78</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">1.78</td>
<td class="gt_row gt_right">1.78</td>
<td class="gt_row gt_right">0.78</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
<td class="gt_row gt_right">-0.22</td>
</tr>
</tbody>
</table>

</div>

``` python
round(raw_iipsc.mean(axis=1, skipna=True), 2)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>


    0    1.00
    1    0.94
    2    1.41
    3    2.29
    4    0.62
    5    0.28
    6    0.50
    7    0.74
    8    0.97
    9    0.22
    dtype: float64

``` python
round(ips_iipsc.mean(axis=1, skipna=True), 2)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>


    0    0.0
    1    0.0
    2    0.0
    3   -0.0
    4    0.0
    5    0.0
    6    0.0
    7    0.0
    8    0.0
    9    0.0
    dtype: float64

### Scoring item-level data

``` python
iipsc.info_scales()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">The IIP-SC contains 8 scales:</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">PA (90°): Domineering</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">BC (135°): Vindictive</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">DE (180°): Cold</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">FG (225°): Socially avoidant</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">HI (270°): Nonassertive</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">JK (315°): Exploitable</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">├── </span><span style="font-weight: bold">LM (360°): Overly nurturant</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">└── </span><span style="font-weight: bold">NO (45°): Intrusive</span>
</pre>

``` python
scale_scores = score(
    data=raw_iipsc, items=np.arange(0, 32), append=False, instrument="iipsc"
)
GT(scale_scores.round(2))
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
<div id="rtsqjersif" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#rtsqjersif table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#rtsqjersif thead, tbody, tfoot, tr, td, th { border-style: none !important; }
 tr { background-color: transparent !important; }
#rtsqjersif p { margin: 0 !important; padding: 0 !important; }
 #rtsqjersif .gt_table { display: table !important; border-collapse: collapse !important; line-height: normal !important; margin-left: auto !important; margin-right: auto !important; color: #333333 !important; font-size: 16px !important; font-weight: normal !important; font-style: normal !important; background-color: #FFFFFF !important; width: auto !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #A8A8A8 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #A8A8A8 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; }
 #rtsqjersif .gt_caption { padding-top: 4px !important; padding-bottom: 4px !important; }
 #rtsqjersif .gt_title { color: #333333 !important; font-size: 125% !important; font-weight: initial !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; border-bottom-color: #FFFFFF !important; border-bottom-width: 0 !important; }
 #rtsqjersif .gt_subtitle { color: #333333 !important; font-size: 85% !important; font-weight: initial !important; padding-top: 3px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; border-top-color: #FFFFFF !important; border-top-width: 0 !important; }
 #rtsqjersif .gt_heading { background-color: #FFFFFF !important; text-align: center !important; border-bottom-color: #FFFFFF !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #rtsqjersif .gt_bottom_border { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #rtsqjersif .gt_col_headings { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #rtsqjersif .gt_col_heading { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; overflow-x: hidden !important; }
 #rtsqjersif .gt_column_spanner_outer { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; padding-top: 0 !important; padding-bottom: 0 !important; padding-left: 4px !important; padding-right: 4px !important; }
 #rtsqjersif .gt_column_spanner_outer:first-child { padding-left: 0 !important; }
 #rtsqjersif .gt_column_spanner_outer:last-child { padding-right: 0 !important; }
 #rtsqjersif .gt_column_spanner { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; overflow-x: hidden !important; display: inline-block !important; width: 100% !important; }
 #rtsqjersif .gt_spanner_row { border-bottom-style: hidden !important; }
 #rtsqjersif .gt_group_heading { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; text-align: left !important; }
 #rtsqjersif .gt_empty_group_heading { padding: 0.5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: middle !important; }
 #rtsqjersif .gt_from_md> :first-child { margin-top: 0 !important; }
 #rtsqjersif .gt_from_md> :last-child { margin-bottom: 0 !important; }
 #rtsqjersif .gt_row { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; margin: 10px !important; border-top-style: solid !important; border-top-width: 1px !important; border-top-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; overflow-x: hidden !important; }
 #rtsqjersif .gt_stub { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; }
 #rtsqjersif .gt_stub_row_group { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; vertical-align: top !important; }
 #rtsqjersif .gt_row_group_first td { border-top-width: 2px !important; }
 #rtsqjersif .gt_row_group_first th { border-top-width: 2px !important; }
 #rtsqjersif .gt_striped { color: #333333 !important; background-color: #F4F4F4 !important; }
 #rtsqjersif .gt_table_body { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #rtsqjersif .gt_grand_summary_row { color: #333333 !important; background-color: #FFFFFF !important; text-transform: inherit !important; padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; }
 #rtsqjersif .gt_first_grand_summary_row_bottom { border-top-style: double !important; border-top-width: 6px !important; border-top-color: #D3D3D3 !important; }
 #rtsqjersif .gt_last_grand_summary_row_top { border-bottom-style: double !important; border-bottom-width: 6px !important; border-bottom-color: #D3D3D3 !important; }
 #rtsqjersif .gt_sourcenotes { color: #333333 !important; background-color: #FFFFFF !important; border-bottom-style: none !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; }
 #rtsqjersif .gt_sourcenote { font-size: 90% !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; text-align: left !important; }
 #rtsqjersif .gt_left { text-align: left !important; }
 #rtsqjersif .gt_center { text-align: center !important; }
 #rtsqjersif .gt_right { text-align: right !important; font-variant-numeric: tabular-nums !important; }
 #rtsqjersif .gt_font_normal { font-weight: normal !important; }
 #rtsqjersif .gt_font_bold { font-weight: bold !important; }
 #rtsqjersif .gt_font_italic { font-style: italic !important; }
 #rtsqjersif .gt_super { font-size: 65% !important; }
 #rtsqjersif .gt_footnote_marks { font-size: 75% !important; vertical-align: 0.4em !important; position: initial !important; }
 #rtsqjersif .gt_asterisk { font-size: 100% !important; vertical-align: 0 !important; }

</style>

<table class="gt_table" data-quarto-postprocess="true"
data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
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
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right">1.75</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">1.25</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.25</td>
<td class="gt_row gt_right">1.5</td>
<td class="gt_row gt_right">0.75</td>
</tr>
<tr>
<td class="gt_row gt_right">0.25</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.25</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">1.75</td>
<td class="gt_row gt_right">1.25</td>
<td class="gt_row gt_right">1.0</td>
</tr>
<tr>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">0.75</td>
<td class="gt_row gt_right">0.75</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">2.25</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">2.5</td>
<td class="gt_row gt_right">2.0</td>
</tr>
<tr>
<td class="gt_row gt_right">1.75</td>
<td class="gt_row gt_right">2.25</td>
<td class="gt_row gt_right">2.5</td>
<td class="gt_row gt_right">2.33</td>
<td class="gt_row gt_right">2.5</td>
<td class="gt_row gt_right">2.0</td>
<td class="gt_row gt_right">2.5</td>
<td class="gt_row gt_right">2.5</td>
</tr>
<tr>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.75</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.25</td>
<td class="gt_row gt_right">1.25</td>
<td class="gt_row gt_right">0.75</td>
</tr>
<tr>
<td class="gt_row gt_right">0.25</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">1.0</td>
</tr>
<tr>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">0.75</td>
<td class="gt_row gt_right">0.25</td>
</tr>
<tr>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">0.25</td>
<td class="gt_row gt_right">0.75</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">0.67</td>
<td class="gt_row gt_right">1.75</td>
<td class="gt_row gt_right">1.0</td>
</tr>
<tr>
<td class="gt_row gt_right">0.75</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">1.5</td>
<td class="gt_row gt_right">0.75</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">2.75</td>
<td class="gt_row gt_right">0.5</td>
</tr>
<tr>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.0</td>
<td class="gt_row gt_right">0.5</td>
<td class="gt_row gt_right">1.0</td>
<td class="gt_row gt_right">0.25</td>
</tr>
</tbody>
</table>

</div>

### Standardizing scale-level data

``` python
iipsc.info_norms()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">The IIP-SC currently has 2 normative data set(s):</span>

1. 872 American college students
   Hopwood, Pincus, DeMoor, &amp; Koonce (2011)
   https://doi.org/10.1080/00223890802388665
2. 106 American psychiatric outpatients
   Soldz, Budman, Demby, &amp; Merry (1995)
   https://doi.org/10.1177/1073191195002001006

</pre>

``` python
z_scales = iipsc.norm_standardize(
    data=scale_scores,
    scales=np.arange(0, 8),
    sample_id=1,
    append=False,
)
GT(z_scales.round(2))
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
<div id="vwgnzichry" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#vwgnzichry table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#vwgnzichry thead, tbody, tfoot, tr, td, th { border-style: none !important; }
 tr { background-color: transparent !important; }
#vwgnzichry p { margin: 0 !important; padding: 0 !important; }
 #vwgnzichry .gt_table { display: table !important; border-collapse: collapse !important; line-height: normal !important; margin-left: auto !important; margin-right: auto !important; color: #333333 !important; font-size: 16px !important; font-weight: normal !important; font-style: normal !important; background-color: #FFFFFF !important; width: auto !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #A8A8A8 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #A8A8A8 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; }
 #vwgnzichry .gt_caption { padding-top: 4px !important; padding-bottom: 4px !important; }
 #vwgnzichry .gt_title { color: #333333 !important; font-size: 125% !important; font-weight: initial !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; border-bottom-color: #FFFFFF !important; border-bottom-width: 0 !important; }
 #vwgnzichry .gt_subtitle { color: #333333 !important; font-size: 85% !important; font-weight: initial !important; padding-top: 3px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; border-top-color: #FFFFFF !important; border-top-width: 0 !important; }
 #vwgnzichry .gt_heading { background-color: #FFFFFF !important; text-align: center !important; border-bottom-color: #FFFFFF !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #vwgnzichry .gt_bottom_border { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #vwgnzichry .gt_col_headings { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #vwgnzichry .gt_col_heading { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; overflow-x: hidden !important; }
 #vwgnzichry .gt_column_spanner_outer { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; padding-top: 0 !important; padding-bottom: 0 !important; padding-left: 4px !important; padding-right: 4px !important; }
 #vwgnzichry .gt_column_spanner_outer:first-child { padding-left: 0 !important; }
 #vwgnzichry .gt_column_spanner_outer:last-child { padding-right: 0 !important; }
 #vwgnzichry .gt_column_spanner { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; overflow-x: hidden !important; display: inline-block !important; width: 100% !important; }
 #vwgnzichry .gt_spanner_row { border-bottom-style: hidden !important; }
 #vwgnzichry .gt_group_heading { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; text-align: left !important; }
 #vwgnzichry .gt_empty_group_heading { padding: 0.5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: middle !important; }
 #vwgnzichry .gt_from_md> :first-child { margin-top: 0 !important; }
 #vwgnzichry .gt_from_md> :last-child { margin-bottom: 0 !important; }
 #vwgnzichry .gt_row { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; margin: 10px !important; border-top-style: solid !important; border-top-width: 1px !important; border-top-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; overflow-x: hidden !important; }
 #vwgnzichry .gt_stub { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; }
 #vwgnzichry .gt_stub_row_group { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; vertical-align: top !important; }
 #vwgnzichry .gt_row_group_first td { border-top-width: 2px !important; }
 #vwgnzichry .gt_row_group_first th { border-top-width: 2px !important; }
 #vwgnzichry .gt_striped { color: #333333 !important; background-color: #F4F4F4 !important; }
 #vwgnzichry .gt_table_body { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #vwgnzichry .gt_grand_summary_row { color: #333333 !important; background-color: #FFFFFF !important; text-transform: inherit !important; padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; }
 #vwgnzichry .gt_first_grand_summary_row_bottom { border-top-style: double !important; border-top-width: 6px !important; border-top-color: #D3D3D3 !important; }
 #vwgnzichry .gt_last_grand_summary_row_top { border-bottom-style: double !important; border-bottom-width: 6px !important; border-bottom-color: #D3D3D3 !important; }
 #vwgnzichry .gt_sourcenotes { color: #333333 !important; background-color: #FFFFFF !important; border-bottom-style: none !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; }
 #vwgnzichry .gt_sourcenote { font-size: 90% !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; text-align: left !important; }
 #vwgnzichry .gt_left { text-align: left !important; }
 #vwgnzichry .gt_center { text-align: center !important; }
 #vwgnzichry .gt_right { text-align: right !important; font-variant-numeric: tabular-nums !important; }
 #vwgnzichry .gt_font_normal { font-weight: normal !important; }
 #vwgnzichry .gt_font_bold { font-weight: bold !important; }
 #vwgnzichry .gt_font_italic { font-style: italic !important; }
 #vwgnzichry .gt_super { font-size: 65% !important; }
 #vwgnzichry .gt_footnote_marks { font-size: 75% !important; vertical-align: 0.4em !important; position: initial !important; }
 #vwgnzichry .gt_asterisk { font-size: 100% !important; vertical-align: 0 !important; }

</style>

<table class="gt_table" data-quarto-postprocess="true"
data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="PA_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">PA_z</th>
<th id="BC_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">BC_z</th>
<th id="DE_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">DE_z</th>
<th id="FG_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">FG_z</th>
<th id="HI_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">HI_z</th>
<th id="JK_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">JK_z</th>
<th id="LM_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">LM_z</th>
<th id="NO_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">NO_z</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right">1.5</td>
<td class="gt_row gt_right">1.75</td>
<td class="gt_row gt_right">0.41</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">-1.01</td>
<td class="gt_row gt_right">-1.33</td>
<td class="gt_row gt_right">0.04</td>
<td class="gt_row gt_right">-0.34</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.77</td>
<td class="gt_row gt_right">-0.42</td>
<td class="gt_row gt_right">-0.76</td>
<td class="gt_row gt_right">-0.58</td>
<td class="gt_row gt_right">0.63</td>
<td class="gt_row gt_right">0.43</td>
<td class="gt_row gt_right">-0.26</td>
<td class="gt_row gt_right">-0.03</td>
</tr>
<tr>
<td class="gt_row gt_right">0.36</td>
<td class="gt_row gt_right">-0.06</td>
<td class="gt_row gt_right">-0.18</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">0.91</td>
<td class="gt_row gt_right">0.72</td>
<td class="gt_row gt_right">1.25</td>
<td class="gt_row gt_right">1.22</td>
</tr>
<tr>
<td class="gt_row gt_right">1.5</td>
<td class="gt_row gt_right">2.11</td>
<td class="gt_row gt_right">1.87</td>
<td class="gt_row gt_right">1.36</td>
<td class="gt_row gt_right">1.18</td>
<td class="gt_row gt_right">0.72</td>
<td class="gt_row gt_right">1.25</td>
<td class="gt_row gt_right">1.84</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.39</td>
<td class="gt_row gt_right">-0.06</td>
<td class="gt_row gt_right">-1.05</td>
<td class="gt_row gt_right">-0.05</td>
<td class="gt_row gt_right">-1.01</td>
<td class="gt_row gt_right">-1.33</td>
<td class="gt_row gt_right">-0.26</td>
<td class="gt_row gt_right">-0.34</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.77</td>
<td class="gt_row gt_right">-1.15</td>
<td class="gt_row gt_right">-1.05</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">-1.55</td>
<td class="gt_row gt_right">-1.62</td>
<td class="gt_row gt_right">-0.56</td>
<td class="gt_row gt_right">-0.03</td>
</tr>
<tr>
<td class="gt_row gt_right">0.36</td>
<td class="gt_row gt_right">-1.15</td>
<td class="gt_row gt_right">-1.05</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">-0.46</td>
<td class="gt_row gt_right">-0.45</td>
<td class="gt_row gt_right">-0.87</td>
<td class="gt_row gt_right">-0.97</td>
</tr>
<tr>
<td class="gt_row gt_right">0.36</td>
<td class="gt_row gt_right">-0.79</td>
<td class="gt_row gt_right">-0.18</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">-1.01</td>
<td class="gt_row gt_right">-0.84</td>
<td class="gt_row gt_right">0.35</td>
<td class="gt_row gt_right">-0.03</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.02</td>
<td class="gt_row gt_right">-0.42</td>
<td class="gt_row gt_right">0.7</td>
<td class="gt_row gt_right">-0.31</td>
<td class="gt_row gt_right">-1.55</td>
<td class="gt_row gt_right">-0.45</td>
<td class="gt_row gt_right">1.56</td>
<td class="gt_row gt_right">-0.66</td>
</tr>
<tr>
<td class="gt_row gt_right">-1.15</td>
<td class="gt_row gt_right">-1.15</td>
<td class="gt_row gt_right">-1.05</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">-1.55</td>
<td class="gt_row gt_right">-1.04</td>
<td class="gt_row gt_right">-0.56</td>
<td class="gt_row gt_right">-0.97</td>
</tr>
</tbody>
</table>

</div>

``` python
norm_df = norm_standardize(
    data=scale_scores,
    instrument="iipsc",
    scales=np.arange(0, 8),
    sample_id=1,
    append=False,
)

GT(norm_df.round(2))
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
<div id="bgoawtdyqk" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#bgoawtdyqk table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#bgoawtdyqk thead, tbody, tfoot, tr, td, th { border-style: none !important; }
 tr { background-color: transparent !important; }
#bgoawtdyqk p { margin: 0 !important; padding: 0 !important; }
 #bgoawtdyqk .gt_table { display: table !important; border-collapse: collapse !important; line-height: normal !important; margin-left: auto !important; margin-right: auto !important; color: #333333 !important; font-size: 16px !important; font-weight: normal !important; font-style: normal !important; background-color: #FFFFFF !important; width: auto !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #A8A8A8 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #A8A8A8 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; }
 #bgoawtdyqk .gt_caption { padding-top: 4px !important; padding-bottom: 4px !important; }
 #bgoawtdyqk .gt_title { color: #333333 !important; font-size: 125% !important; font-weight: initial !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; border-bottom-color: #FFFFFF !important; border-bottom-width: 0 !important; }
 #bgoawtdyqk .gt_subtitle { color: #333333 !important; font-size: 85% !important; font-weight: initial !important; padding-top: 3px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; border-top-color: #FFFFFF !important; border-top-width: 0 !important; }
 #bgoawtdyqk .gt_heading { background-color: #FFFFFF !important; text-align: center !important; border-bottom-color: #FFFFFF !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #bgoawtdyqk .gt_bottom_border { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #bgoawtdyqk .gt_col_headings { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; }
 #bgoawtdyqk .gt_col_heading { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; padding-left: 5px !important; padding-right: 5px !important; overflow-x: hidden !important; }
 #bgoawtdyqk .gt_column_spanner_outer { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: normal !important; text-transform: inherit !important; padding-top: 0 !important; padding-bottom: 0 !important; padding-left: 4px !important; padding-right: 4px !important; }
 #bgoawtdyqk .gt_column_spanner_outer:first-child { padding-left: 0 !important; }
 #bgoawtdyqk .gt_column_spanner_outer:last-child { padding-right: 0 !important; }
 #bgoawtdyqk .gt_column_spanner { border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: bottom !important; padding-top: 5px !important; padding-bottom: 5px !important; overflow-x: hidden !important; display: inline-block !important; width: 100% !important; }
 #bgoawtdyqk .gt_spanner_row { border-bottom-style: hidden !important; }
 #bgoawtdyqk .gt_group_heading { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; text-align: left !important; }
 #bgoawtdyqk .gt_empty_group_heading { padding: 0.5px !important; color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; vertical-align: middle !important; }
 #bgoawtdyqk .gt_from_md> :first-child { margin-top: 0 !important; }
 #bgoawtdyqk .gt_from_md> :last-child { margin-bottom: 0 !important; }
 #bgoawtdyqk .gt_row { padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; margin: 10px !important; border-top-style: solid !important; border-top-width: 1px !important; border-top-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 1px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 1px !important; border-right-color: #D3D3D3 !important; vertical-align: middle !important; overflow-x: hidden !important; }
 #bgoawtdyqk .gt_stub { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; }
 #bgoawtdyqk .gt_stub_row_group { color: #333333 !important; background-color: #FFFFFF !important; font-size: 100% !important; font-weight: initial !important; text-transform: inherit !important; border-right-style: solid !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; padding-left: 5px !important; padding-right: 5px !important; vertical-align: top !important; }
 #bgoawtdyqk .gt_row_group_first td { border-top-width: 2px !important; }
 #bgoawtdyqk .gt_row_group_first th { border-top-width: 2px !important; }
 #bgoawtdyqk .gt_striped { color: #333333 !important; background-color: #F4F4F4 !important; }
 #bgoawtdyqk .gt_table_body { border-top-style: solid !important; border-top-width: 2px !important; border-top-color: #D3D3D3 !important; border-bottom-style: solid !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; }
 #bgoawtdyqk .gt_grand_summary_row { color: #333333 !important; background-color: #FFFFFF !important; text-transform: inherit !important; padding-top: 8px !important; padding-bottom: 8px !important; padding-left: 5px !important; padding-right: 5px !important; }
 #bgoawtdyqk .gt_first_grand_summary_row_bottom { border-top-style: double !important; border-top-width: 6px !important; border-top-color: #D3D3D3 !important; }
 #bgoawtdyqk .gt_last_grand_summary_row_top { border-bottom-style: double !important; border-bottom-width: 6px !important; border-bottom-color: #D3D3D3 !important; }
 #bgoawtdyqk .gt_sourcenotes { color: #333333 !important; background-color: #FFFFFF !important; border-bottom-style: none !important; border-bottom-width: 2px !important; border-bottom-color: #D3D3D3 !important; border-left-style: none !important; border-left-width: 2px !important; border-left-color: #D3D3D3 !important; border-right-style: none !important; border-right-width: 2px !important; border-right-color: #D3D3D3 !important; }
 #bgoawtdyqk .gt_sourcenote { font-size: 90% !important; padding-top: 4px !important; padding-bottom: 4px !important; padding-left: 5px !important; padding-right: 5px !important; text-align: left !important; }
 #bgoawtdyqk .gt_left { text-align: left !important; }
 #bgoawtdyqk .gt_center { text-align: center !important; }
 #bgoawtdyqk .gt_right { text-align: right !important; font-variant-numeric: tabular-nums !important; }
 #bgoawtdyqk .gt_font_normal { font-weight: normal !important; }
 #bgoawtdyqk .gt_font_bold { font-weight: bold !important; }
 #bgoawtdyqk .gt_font_italic { font-style: italic !important; }
 #bgoawtdyqk .gt_super { font-size: 65% !important; }
 #bgoawtdyqk .gt_footnote_marks { font-size: 75% !important; vertical-align: 0.4em !important; position: initial !important; }
 #bgoawtdyqk .gt_asterisk { font-size: 100% !important; vertical-align: 0 !important; }

</style>

<table class="gt_table" data-quarto-postprocess="true"
data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="PA_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">PA_z</th>
<th id="BC_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">BC_z</th>
<th id="DE_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">DE_z</th>
<th id="FG_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">FG_z</th>
<th id="HI_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">HI_z</th>
<th id="JK_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">JK_z</th>
<th id="LM_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">LM_z</th>
<th id="NO_z" class="gt_col_heading gt_columns_bottom_border gt_right"
data-quarto-table-cell-role="th" scope="col">NO_z</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right">1.5</td>
<td class="gt_row gt_right">1.75</td>
<td class="gt_row gt_right">0.41</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">-1.01</td>
<td class="gt_row gt_right">-1.33</td>
<td class="gt_row gt_right">0.04</td>
<td class="gt_row gt_right">-0.34</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.77</td>
<td class="gt_row gt_right">-0.42</td>
<td class="gt_row gt_right">-0.76</td>
<td class="gt_row gt_right">-0.58</td>
<td class="gt_row gt_right">0.63</td>
<td class="gt_row gt_right">0.43</td>
<td class="gt_row gt_right">-0.26</td>
<td class="gt_row gt_right">-0.03</td>
</tr>
<tr>
<td class="gt_row gt_right">0.36</td>
<td class="gt_row gt_right">-0.06</td>
<td class="gt_row gt_right">-0.18</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">0.91</td>
<td class="gt_row gt_right">0.72</td>
<td class="gt_row gt_right">1.25</td>
<td class="gt_row gt_right">1.22</td>
</tr>
<tr>
<td class="gt_row gt_right">1.5</td>
<td class="gt_row gt_right">2.11</td>
<td class="gt_row gt_right">1.87</td>
<td class="gt_row gt_right">1.36</td>
<td class="gt_row gt_right">1.18</td>
<td class="gt_row gt_right">0.72</td>
<td class="gt_row gt_right">1.25</td>
<td class="gt_row gt_right">1.84</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.39</td>
<td class="gt_row gt_right">-0.06</td>
<td class="gt_row gt_right">-1.05</td>
<td class="gt_row gt_right">-0.05</td>
<td class="gt_row gt_right">-1.01</td>
<td class="gt_row gt_right">-1.33</td>
<td class="gt_row gt_right">-0.26</td>
<td class="gt_row gt_right">-0.34</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.77</td>
<td class="gt_row gt_right">-1.15</td>
<td class="gt_row gt_right">-1.05</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">-1.55</td>
<td class="gt_row gt_right">-1.62</td>
<td class="gt_row gt_right">-0.56</td>
<td class="gt_row gt_right">-0.03</td>
</tr>
<tr>
<td class="gt_row gt_right">0.36</td>
<td class="gt_row gt_right">-1.15</td>
<td class="gt_row gt_right">-1.05</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">-0.46</td>
<td class="gt_row gt_right">-0.45</td>
<td class="gt_row gt_right">-0.87</td>
<td class="gt_row gt_right">-0.97</td>
</tr>
<tr>
<td class="gt_row gt_right">0.36</td>
<td class="gt_row gt_right">-0.79</td>
<td class="gt_row gt_right">-0.18</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">-1.01</td>
<td class="gt_row gt_right">-0.84</td>
<td class="gt_row gt_right">0.35</td>
<td class="gt_row gt_right">-0.03</td>
</tr>
<tr>
<td class="gt_row gt_right">-0.02</td>
<td class="gt_row gt_right">-0.42</td>
<td class="gt_row gt_right">0.7</td>
<td class="gt_row gt_right">-0.31</td>
<td class="gt_row gt_right">-1.55</td>
<td class="gt_row gt_right">-0.45</td>
<td class="gt_row gt_right">1.56</td>
<td class="gt_row gt_right">-0.66</td>
</tr>
<tr>
<td class="gt_row gt_right">-1.15</td>
<td class="gt_row gt_right">-1.15</td>
<td class="gt_row gt_right">-1.05</td>
<td class="gt_row gt_right">-1.11</td>
<td class="gt_row gt_right">-1.55</td>
<td class="gt_row gt_right">-1.04</td>
<td class="gt_row gt_right">-0.56</td>
<td class="gt_row gt_right">-0.97</td>
</tr>
</tbody>
</table>

</div>
