"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { SearchIndex, type SearchResult } from "@/lib/search";
import { useStore } from "@/lib/store";
import RouteBullet, { RouteBullets } from "./RouteBullet";
import styles from "./SearchBox.module.css";

/** Station + route search behind an icon (`/` opens it); choosing a result flies the camera there. */
export default function SearchBox() {
  const net = useStore((s) => s.network);
  const requestCamera = useStore((s) => s.requestCamera);
  const setHighlight = useStore((s) => s.setHighlight);
  const index = useMemo(() => (net ? new SearchIndex(net) : null), [net]);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const expand = () => {
    setExpanded(true);
    requestAnimationFrame(() => inputRef.current?.focus());
  };
  const collapse = () => {
    setExpanded(false);
    setOpen(false);
    setQuery("");
  };
  const listId = useId();
  const results = useMemo(() => (index && query.trim() ? index.query(query, 8) : []), [index, query]);

  // "/" focuses search from anywhere (unless typing in a field).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      e.preventDefault();
      setExpanded(true);
      requestAnimationFrame(() => inputRef.current?.focus());
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const choose = (r: SearchResult) => {
    if (r.kind === "station") {
      requestCamera({ kind: "view", view: { cx: r.x, cy: r.y, mPerPx: 3 } });
      setHighlight(r.stationIdx, 3200);
    } else {
      requestCamera({ kind: "bbox", bbox: r.bbox });
    }
    inputRef.current?.blur();
    collapse();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((a) => Math.min(results.length - 1, a + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(0, a - 1));
    } else if (e.key === "Enter") {
      const r = results[active] ?? results[0];
      if (r) {
        e.preventDefault();
        choose(r);
      }
    } else if (e.key === "Escape") {
      inputRef.current?.blur();
      collapse();
    }
  };

  const showList = open && results.length > 0;
  return (
    <div className={styles.root} role="search" data-expanded={expanded}>
      <button
        type="button"
        className={styles.toggle}
        aria-label="Search stations and lines"
        aria-expanded={expanded}
        title="Search stations and lines (/)"
        onClick={() => (expanded ? collapse() : expand())}
        tabIndex={expanded ? -1 : 0}
      >
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <circle cx="8.6" cy="8.6" r="5.4" fill="none" stroke="currentColor" strokeWidth="1.7" />
          <path d="M12.8 12.8l4.2 4.2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
        </svg>
      </button>
      <label className={styles.field} aria-hidden={!expanded}>
        <svg className={styles.icon} viewBox="0 0 20 20" aria-hidden="true">
          <circle cx="8.6" cy="8.6" r="5.4" fill="none" stroke="currentColor" strokeWidth="1.7" />
          <path d="M12.8 12.8l4.2 4.2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
        </svg>
        <span className="visually-hidden">Search stations and lines</span>
        <input
          ref={inputRef}
          className={styles.input}
          type="search"
          inputMode="search"
          autoComplete="off"
          spellCheck={false}
          placeholder={net ? "Search a station or line…" : "Loading stations…"}
          disabled={!net}
          value={query}
          role="combobox"
          aria-expanded={showList}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={showList ? `${listId}-${active}` : undefined}
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
            setOpen(true);
          }}
          tabIndex={expanded ? 0 : -1}
          onFocus={() => {
            setExpanded(true);
            setOpen(true);
          }}
          onBlur={(e) => {
            const empty = e.currentTarget.value.trim() === "";
            setTimeout(() => {
              setOpen(false);
              if (empty) setExpanded(false);
            }, 120);
          }}
          onKeyDown={onKeyDown}
        />
        <kbd className={styles.kbd} aria-hidden="true">
          /
        </kbd>
      </label>
      {showList && (
        <ul id={listId} className={styles.list} role="listbox">
          {results.map((r, i) => (
            <li
              key={r.key}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              className={i === active ? styles.itemActive : styles.item}
              onMouseDown={(e) => e.preventDefault()}
              onMouseEnter={() => setActive(i)}
              onClick={() => choose(r)}
            >
              {r.kind === "station" ? (
                <>
                  <span className={styles.name}>{r.name}</span>
                  <RouteBullets routes={r.routes} size={15} max={9} />
                </>
              ) : (
                <>
                  <span className={styles.name}>
                    <RouteBullet routeIdx={r.routeIdx} size={16} /> {r.name}
                  </span>
                  <span className={styles.kind}>Line</span>
                </>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
