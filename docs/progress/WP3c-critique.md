# WP3c critique: MTA-themed reskin (design review)

Reviewer: product-design pass, judged only on what a user sees (no source read). 2026-11-22.

**How it was browsed.** The user's :3000 / :8000 servers were not running, so nothing was stopped or restarted on those ports. Following the override in `WP3c.md`, I ran temporary servers for the session:
- backend on :8010 with the current champion, `runs/pertrain_global/export_best`;
- `next dev` on :3200.

Both are stopped now. Playwright script: `scratchpad/evaltools/crit3c.mjs`; screenshots are in `scratchpad/crit3c/`.

**What I covered:**
- desktop 1440×900, citywide and zoomed in, Normal mode;
- the map key;
- the "Break the city" menu;
- "Close 14 St–Union Sq" with AI on for 30 s at 1 min/s, then 30 s at 5 min/s;
- a scoreboard close-up;
- service-change toasts and the dispatch-log drawer;
- the info modal and its tabs;
- keyboard focus;
- laptop 1280×800 with the QBL preset and AI at 5 min/s;
- mobile 390×844: city, menu, AI on, scoreboard, log and info.

There were 0 console errors. Rendering held 60 fps (p95 16.8 ms) with AI on at 5 min/s, citywide and zoomed. Mobile had no horizontal scroll.

## Verdict

**Approve with fixes. The theme lands, but the copy and the LED digits need another pass before shipping.**

Zoomed in, this is unmistakably the NYC subway, and mostly in good taste:
- the black top band with its thin white rule;
- real route bullets in the menu and on the posters;
- saturated ribbons on cream land, white express circles, black local dots;
- the poster-style "SERVICE CHANGE" toast with a big bullet;
- the amber arrival board.

It doesn't feel like a costume. There is no MTA logo. The not-affiliated line appears as a corner credit on desktop and laptop and under the info-modal title.

Four things hold it back:
1. **Some service-change headlines say the wrong place**, which is an honesty problem.
2. **The dot-matrix digits are hard to read**: the decimal point looks like a "+" and 0 looks like a diamond.
3. **The citywide view loses the printed-map look.** Lines look pastel and trains turn them into dashes.
4. **Express/local symbols are wrong at many local stations**, so a map convention we added now misinforms.

Below those come collisions between the credit, the drawer and the map, and a signage rule applied so widely that it reads as a bevel.

---

## Issues (most important first)

### 1. Gap-train headlines name the wrong location (high)
- **Where:** AI service-change toasts and dispatch-log entries for "Send a gap train".
- **Problem:** The bold headline and the body disagree, sometimes by many miles. Seen live:
  - "Extra train enters service near **42 St-Port Authority Bus Terminal**", with the body "out of Jamaica Yard — it enters service as an E train at **Forest Hills-71 Av**";
  - "…near **Grand St**" for a B train entering at **Bedford Park Blvd** from Concourse Yard;
  - "…near **Nostrand Av**" for an A train entering at **Euclid Av**.

  The headline is the line people read, and here it is false. The headline seems to name the gap the train is filling. The poster idiom reads it as where the service change happens. In a project whose hard rule is honesty, this matters most.
- **Fix:** Make the headline the observable fact: "Extra **E** train enters service at Forest Hills-71 Av". Put the why on the reason line: "Fills a gap near 42 St-Port Authority · from Jamaica Yard · in ~7 min". Add a unit test asserting that the headline station equals the entry station.

### 2. LED digits are hard to read and partly misread (high)
- **Where:** Scoreboard values and deltas, the "since AI took over" timer, "Rider-min waited", the mobile "Wait 2.8 min" pill, the closed-station countdown.
- **Problem:** At this size, Doto shows the decimal point as a small cross. "5.3 min" reads as "5+3 min", "−1.4%" as "−1+4%" and "3.49M" as "3+49M". The zero is a sparse diamond (◊) rather than a 5×7 rounded zero.

  "±" isn't in the font, so a heavy Helvetica "±" sits next to dotted digits (for example "±0", "±0%"). The grey delta digits ("+233", "−672") are thin dots on black, well below the legibility of the white labels. The lowercase dot-matrix "min" on mobile reads as "rnin".

  Real countdown clocks use a dense 5×7 matrix where every glyph is unambiguous.
- **Fix:**
  - Use a denser setting: Doto at a heavier weight/roundness, or DotGothic16 for digits.
  - Render the decimal point as a real 2×2-dot full stop. If needed, wrap "." in a span with its own glyph or font.
  - Check that "0" vs "8" and "." vs "+" are distinguishable at 18 px.
  - Draw "±" in the LED font, or replace it with a dotted "—" or "0".
  - Set the unit ("min") in white Helvetica rather than LED.
  - Raise grey deltas to at least 4.5:1, or show them in amber at 60%.

### 3. Skip-stop posters don't name the skipped station (medium)
- **Where:** Toasts and log headlines for "Skip the next stop".
- **Problem:** "Skips the next stop after Bowling Green" hides the one fact a rider needs (Wall St is bypassed). That fact appears only in the hover title and the log body. The brief's own example names the stations ("Trains run local · 125 St → Grand Central"). Real service-change posters lead with "No trains stop at …".
- **Fix:** Use the headline "**No stop at Wall St**" with "Bowling Green → Fulton St · 14 riders ride on" underneath. Apply the same pattern to holds ("Held at 23 St-Baruch College · 14 St-Union Sq ahead is closed", which is already good) and short-turns ("Turns back at Bowling Green", also good).

### 4. Express/local symbols are wrong at many local stations (medium)
- **Where:** Map, zoomed in on Queens.
- **Problem:** Several local-only stations draw as white circles, the express-stop symbol:
  - Steinway St, 46 St and Northern Blvd on the QBL local;
  - 33 St-Rawson St, 40 St-Lowery St, 46 St-Bliss St, 52 St and 103 St-Corona Plaza on the 7 local.

  Correct cases include 61 St-Woodside, 74 St, Junction Blvd and Queensboro Plaza. The progress log calls this a heuristic. To anyone who knows the map, it is visibly wrong, and it undermines the convention the reskin borrowed.
- **Fix:** Derive it from each route's stopping pattern. The sim already knows which trains skip which stations, because skip and express logic depend on it. A station is a white circle only if an express pattern stops there or it is a transfer. If the data is missing, default to a black dot: a wrong black dot is less misleading than a wrong white circle.

### 5. The citywide view doesn't look like the printed map (medium)
- **Where:** Desktop and mobile citywide, the first screen everyone sees.
- **Problem:** At citywide zoom the ribbons render washed out. Sampled pixels against the official colors:

  | Line | Sampled | Official |
  |---|---|---|
  | Red | `#E15764` | `#EE352E` |
  | Blue | `#5D90C7` | `#0039A6` |
  | Orange | `#F08C3D` | `#FF6319` |
  | Brown | `#AC8F75` | `#996633` |

  Trains are almost as wide as the line, so every line becomes a dashed "sausage chain". Manhattan becomes confetti. The printed map's signature is saturated, solid, parallel ribbons, and that only appears once you zoom in. At citywide the local dots and transfer connectors disappear entirely.
- **Fix:**
  - Draw ribbons at full opacity in official colors at every zoom.
  - At citywide, show trains as small dots inset on the ribbon (about 60% of ribbon width), or hide healthy trains below a zoom threshold and show only stuck, held or AI-touched ones.
  - Thicken ribbons slightly at low zoom so trunk bundles (Lex, 8 Av, Broadway) read as bundles.

### 6. Status cues collide with route colors (medium)
- **Where:** Map, AI mode.
- **Problem:**
  - The "AI just acted" cue is a thin, light-blue ring. On the A/C/E, and on the Queens Blvd bundle under the QBL preset, it is nearly invisible.
  - The red "stuck" glow sits on the 1/2/3 red; the amber "held" ring sits on the N/Q/R/W yellow.

  The light map made this worse than on the dark WP3b map.
- **Fix:** Use cues that don't depend on route color:
  - AI: a black-and-white double ring, or a small black "AI" tab on the pill;
  - held: a white ring with a black outline plus a pause tick;
  - stuck: a red ring with a white gap halo and the pulse.

  Check each cue on blue, red and yellow lines.

### 7. Unlabeled dark-red blocks in the sparklines (medium)
- **Where:** Scoreboard sparklines, desktop and mobile; also in Normal mode during a preset.
- **Problem:** A dark maroon rectangle with a red dashed edge fills part or all of each sparkline. Nothing says what it is (presumably the disruption window). On the mobile board it is the most prominent thing in each row and reads as an error state. It even appears in the "Overcrowded" row, where both lines are flat at 0.
- **Fix:** If it marks the disruption, use a neutral grey hatch with a tiny "closed" tick label, or a thin red baseline segment instead of a filled block. Otherwise remove it. Explain it in the map key or the board footer.

### 8. Credit, drawer and map collide (medium)
- **Where:** Bottom-right on desktop and laptop; the log drawer on all sizes.
- **Problem:**
  - The "Fan-made simulator · Not affiliated with the MTA" credit has no backing. It overprints map content: "Forest Hills-71 Av" on laptop, the G line and "Bedford Av" on desktop.
  - The open dispatch-log drawer covers the bottom of its own "Dispatch log" button and runs into the credit.
  - On mobile, the drawer slides under the translucent dock and the Break-the-city button, so entry text ("Held at 23 St-Baruch C…", "SERVICE CHANGE 8:07 AM") shows through the controls.
- **Fix:**
  - Give the credit a cream chip or halo, or set it inside the map-tools sign as a second line.
  - Anchor the drawer below the button with an 8 px gap, and cap its height at the space above the dock.
  - Make the dock fully opaque, and on mobile hide the floating buttons while the drawer is open.

### 9. The signage rule reads as a bevel (medium)
- **Where:** Scoreboard, log button, menu, dock, map tools, map-key toggle, key panel, drawer, Break-the-city button.
- **Problem:** The thin white rule is the strongest signage cue, but it appears on 8+ small floating cards. Each has 8–10 px rounded corners that the rule bends around, which is especially visible on mobile. On a real platform sign the rule runs straight across a full-width black band with square corners. Repeated on every button, it reads as a UI bevel and stops meaning anything.
- **Fix:**
  - Keep the rule on the top band, the scoreboard, the Break-the-city menu and the poster headers.
  - Drop it from small buttons (log button, map tools, key toggle, Break-the-city trigger).
  - Use a 0–2 px corner radius on the elements that keep it.

### 10. "SERVICE CHANGE" is attached to every action (medium)
- **Where:** Toasts and the dispatch log, which held about 3,000–4,000 entries after a few sim hours.
- **Problem:** Every 30-second hold, skip and switch is a "SERVICE CHANGE" poster. In MTA vocabulary that header means a planned or significant reroute. Repeated thousands of times it loses force and makes the log monotonous. Only the disruption entries get their own headers ("SERVICE ALERT", "SERVICE RESTORED").
- **Fix:** Map each action to the right header, keeping the same poster frame:

  | Action | Header |
  |---|---|
  | Hold | HOLDING |
  | Skip | SKIP STOP |
  | Switch to local / reroute / short-turn | SERVICE CHANGE |
  | Gap train | EXTRA SERVICE |

  Reserve the black bar for the big ones and give the small ones a lighter grey tab.

### 11. "Riders served" label truncates (low)
- **Where:** Scoreboard, desktop and laptop, in both modes once the value passes 1,000,000.
- **Problem:** At 5 min/s this happens about 1.5 sim hours in: "Riders se… 1,392,012", "Riders ser… 2,396,801".
- **Fix:** Show values of 1,000,000 or more as "1.39M" in the LED field, or shorten the label to "Served".

### 12. Ghosted station labels (low)
- **Where:** Map at mid zoom, for example "3 Av", "28 St", "18 St", "Astor Pl", "2 Av", "103 St-Corona Plaza" and "46 St-Bliss St".
- **Problem:** Some labels fade to about 30% opacity, presumably when a train overlaps them, and flicker as trains pass. That fails AA and looks broken.
- **Fix:** Hide colliding labels outright, or keep them at full black with the cream halo and draw the label layer above trains.

### 13. The info modal is the least themed surface (low)
- **Where:** "How it works" modal.
- **Problem:** It is a generic dark card with a blue "HOW IT WORKS" eyebrow, blue list numerals and a navy "NO TRICKS" box. The navy tint survived the reskin. The not-affiliated line is present and well placed.
- **Fix:** Give it a black header band with the rule and the title in white sign type. Make the body a white notice slip, or plain black. Use black or grey for the callout instead of navy, and white numerals.

### 14. Cryptic "6 in 10 s" chip on toasts (low)
- **Where:** Service-change toasts.
- **Problem:** It isn't clear what "6 in 10 s" counts.
- **Fix:** Use "+5 more changes" or "6 changes · last 10 s", or drop it in favor of the log count.

### 15. Toast stack is misaligned (low)
- **Where:** Bottom-center notice stack.
- **Problem:** The "NOTICE" slip (e.g. "14 St-Union Sq reopened · 8:30 AM") and the SERVICE CHANGE poster have different widths and left edges.
- **Fix:** Give them one shared width and left edge.

### 16. Log slips lose their poster frame (low)
- **Where:** Dispatch-log drawer.
- **Problem:** Each entry's black "SERVICE CHANGE" bar sits on the black drawer, so it reads as a floating heading over a white card rather than one poster.
- **Fix:** Use a dark-grey drawer background (about `#1C1C1C`), or give each slip a 1 px light frame that wraps both the bar and the body.

### 17. Mobile scoreboard has a double header (low)
- **Where:** Mobile, expanded board.
- **Problem:** "vs Normal ahead on 1 of 4 ⌃" is followed immediately by "AI VS NORMAL · since AI took over 12m 40s". This duplicates the header and wastes height on a small screen.
- **Fix:** When the board is expanded, turn the collapsed pill into the board header.

### 18. No visible not-affiliated note on mobile (low)
- **Where:** Mobile, where the credit is hidden below 960 px.
- **Problem:** The note appears only in the info modal. The brief asks for an unobtrusive corner credit.
- **Fix:** Add one quiet line at the foot of the Break-the-city sheet or the expanded board.

### 19. Map key mismatches (low)
- **Where:** Map key.
- **Problem:**
  - It uses the British spelling "colour" in an NYC product.
  - There is no swatch for the white transfer connector.
  - The "Closed" swatch (red circle with ×) doesn't match the closed badge on the map (black capsule, "● CLOSED 27:20").
- **Fix:** Use "color", add a connector swatch, and show the badge as drawn.

### 20. Transfer connectors look like wires (low)
- **Where:** Map, zoomed in, for example Court Sq/Court Sq-23 St and 14 St/6 Av.
- **Problem:** Connectors are thin double-stroked lines that form triangles and overprint labels ("Court Sq", "14 St"). The printed map uses a fat white pill with a black outline, about as wide as the station circle.
- **Fix:** Make the connector at least as wide as the station marker, draw it under labels, and join multi-station complexes as one pill where possible.

### 21. Menu sublabel can be misread as a route (low)
- **Where:** "Break the city" menu.
- **Problem:** "Signal problems at Times Sq–42 St · ① ② ③ 7 Av". Right after the bullets, "7 Av" reads like the 7 route.
- **Fix:** Use "Broadway–7 Av Line", or add the word "Line" to every preset sublabel ("Lexington Av Line", "Queens Blvd Line", …).

---

## What works and should stay
- The full-width black top band with the rule, white Helvetica title and clock. It reads as a station sign.
- Route bullets in the preset menu, with correct colors and black letters on N/Q/R/W yellow.
- The poster toast: black "SERVICE CHANGE" bar, "AI Dispatcher" byline, large bullet, plain-language line. "Held at 23 St-Baruch College / 14 St-Union Sq ahead is closed" is exactly right.
- Scoreboard row bullets are pictograms, never letters, so nothing passes for a fake route.
- The scoreboard's honest semantics are intact: since-takeover verdict, grey dead band, twin line.
- Zoomed-in map: saturated ribbons, light gaps between parallel lines, a legible black Helvetica label hierarchy, and train pills that pop with outline and shadow.
- No MTA logo. The not-affiliated wording appears in the info modal ("Not affiliated with or endorsed by the MTA") and as the desktop corner credit.
- Motion is smooth (60 fps), the focus ring is visible, and mobile has no horizontal scroll.
