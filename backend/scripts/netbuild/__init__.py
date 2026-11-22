"""Internal modules of the NYC network build (ARCHITECTURE §2).

Pipeline overview (``scripts/build_nyc_network.py`` drives it through ``artifacts``):

``sources``    fetch GTFS / complexes / stations / ridership / basemap with cache + fallbacks (``http``)
``gtfs``       parse the weekday GTFS feed into trips, stations and shapes
``patterns``   stop patterns, orientation, elementary-hop decomposition
``corridors``  resolve the curated CBTC / middle-track / 3-track ranges of ``infrastructure.json`` (``infra``)
``tracks``     overtake evidence, express (X) tracks, per-pattern track classes
``topology``   track nodes, run/crossover/turnback/yard edges, platforms, segments, interlockings
``geometry``   per-segment polylines cut from shapes; the borough basemap (``geo``)
``timing``     de-quantized run times and speed limits
``places``     station metadata and complexes
``service``    stop counts, headway tables, 3-track peak directions
``demand``     ridership profiles, platform rates, attraction, capacity, hubs, label ranks
``presets``    resolve the curated meltdown presets to indices
``pipeline``   all of the above in order -> ``NetworkModel``
``assemble``   ``network.json`` and ``network_geo.json`` documents (``jsonio``: deterministic JSON)
``validate``   §2.6/§2.7 invariants shared by ``--check`` and the tests
``maneuvers``  dispatch maneuvers available upstream of each preset disruption
``artifacts``  build / validate / serialize / manifest (used by the CLI and the tests)

Everything here uses only the standard library and NumPy.
"""
