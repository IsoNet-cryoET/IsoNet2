import React, { useEffect, useRef, useState } from "react";
import {
    Box,
    Drawer,
    Typography,
    Divider,
    TextField,
    FormControl,
    FormControlLabel,
    Switch,
    InputLabel,
    Select,
    MenuItem,
    Button,
    Accordion,
    AccordionSummary,
    AccordionDetails
} from '@mui/material'

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkSlug from "remark-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import DocsSidebar from "./DocsSidebar";
import introMd from "../../docs/intro.md?raw";
// ---- helpers: detect the correct scroll container ----
function getScrollContainer(el) {
  // find nearest ancestor that can scroll vertically
  let node = el;
  while (node && node !== document.body) {
    const style = getComputedStyle(node);
    const canScroll =
      (style.overflowY === "auto" || style.overflowY === "scroll") &&
      node.scrollHeight > node.clientHeight;
    if (canScroll) return node;
    node = node.parentElement;
  }
  // fallback to window/document
  return document.scrollingElement || document.documentElement;
}

export default function DocumentationPage() {
  const content = introMd;

  const articleRef = useRef(null);          // <div class="docs-article">
  const scrollContainerRef = useRef(null);  // resolved on mount
  const [toc, setToc] = useState([]);
  const [activeId, setActiveId] = useState(null);

  // cache heading positions relative to scroll container
  const headingMetaRef = useRef([]); // [{id, level, top}]

  // 0) Resolve the scroll container once article is mounted
  useEffect(() => {
    if (!articleRef.current) return;
    const container = getScrollContainer(articleRef.current);
    scrollContainerRef.current = container;
  }, []);

  // 1) Build TOC after markdown renders (and whenever content changes)
  useEffect(() => {
    if (!articleRef.current) return;
    const buildTOC = () => {
      const headings = Array.from(
        articleRef.current.querySelectorAll("h1, h2, h3")
      );
      setToc(
        headings.map((el) => ({
          id: el.id,
          text: el.innerText.trim(),
          level: Number(el.tagName.substring(1)),
        }))
      );
    };

    // run now and once layout settles
    buildTOC();
    const raf = requestAnimationFrame(buildTOC);
    const t = setTimeout(buildTOC, 200);

    // in case markdown mutates (images load / fonts swap), observe the article
    const mo = new MutationObserver(buildTOC);
    mo.observe(articleRef.current, { childList: true, subtree: true });

    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(t);
      mo.disconnect();
    };
  }, [content]);

  // 2) Compute heading positions RELATIVE to the scroll container
  useEffect(() => {
    if (!articleRef.current || !scrollContainerRef.current) return;

    const container = scrollContainerRef.current;

    const computePositions = () => {
      const headings = Array.from(
        articleRef.current.querySelectorAll("h1, h2, h3")
      ).filter((el) => el.id);

      if (!headings.length) {
        headingMetaRef.current = [];
        return;
      }

      // container metrics
      const cRect =
        container === document.documentElement || container === document.scrollingElement
          ? { top: 0 }
          : container.getBoundingClientRect();

      headingMetaRef.current = headings.map((el) => {
        const r = el.getBoundingClientRect();
        // position relative to container's scrollTop
        const top =
          (r.top - cRect.top) +
          (container === document.documentElement || container === document.scrollingElement
            ? window.pageYOffset || document.documentElement.scrollTop || 0
            : container.scrollTop);
        return {
          id: el.id,
          level: Number(el.tagName.substring(1)),
          top,
        };
      });
    };

    // initial + after layout settles + on resize + on image load
    computePositions();
    const raf = requestAnimationFrame(computePositions);
    const t = setTimeout(computePositions, 200);

    const onResize = () => computePositions();
    window.addEventListener("resize", onResize);

    const imgs = Array.from(articleRef.current.querySelectorAll("img"));
    const onImgLoad = () => computePositions();
    imgs.forEach((img) => img.addEventListener("load", onImgLoad));

    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(t);
      window.removeEventListener("resize", onResize);
      imgs.forEach((img) => img.removeEventListener("load", onImgLoad));
    };
  }, [content]);

  // 3) Scroll-spy on the CORRECT container
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const TOP_OFFSET = 72; // must match CSS scroll-margin-top
    let ticking = false;

    const getScrollTop = () => {
      return container === document.documentElement || container === document.scrollingElement
        ? (window.pageYOffset || document.documentElement.scrollTop || 0)
        : container.scrollTop;
    };

    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        ticking = false;

        const list = headingMetaRef.current;
        if (!list.length) return;

        const pos = getScrollTop() + TOP_OFFSET + 1;

        // find the last heading whose top <= pos
        let idx = -1;
        for (let i = 0; i < list.length; i++) {
          if (list[i].top <= pos) idx = i;
          else break;
        }
        if (idx < 0) idx = 0;

        const cur = list[idx].id;
        setActiveId((prev) => (prev !== cur ? cur : prev));
      });
    };

    // init + listen
    onScroll();

    const target =
      container === document.documentElement || container === document.scrollingElement
        ? window
        : container;

    target.addEventListener("scroll", onScroll, { passive: true });
    return () => target.removeEventListener("scroll", onScroll);
  }, []);

  // 4) Smooth scroll ON THE CORRECT CONTAINER
  const handleLinkClick = (id) => {
    const container = scrollContainerRef.current;
    if (!container || !articleRef.current) return;

    const target = articleRef.current.querySelector(`#${CSS.escape(id)}`);
    if (!target) return;

    const TOP_OFFSET = 72;

    // compute target Y relative to container
    const cRect =
      container === document.documentElement || container === document.scrollingElement
        ? { top: 0 }
        : container.getBoundingClientRect();

    const tRect = target.getBoundingClientRect();
    const currentScrollTop =
      container === document.documentElement || container === document.scrollingElement
        ? (window.pageYOffset || document.documentElement.scrollTop || 0)
        : container.scrollTop;

    const y = (tRect.top - cRect.top) + currentScrollTop - TOP_OFFSET;

    if (container === document.documentElement || container === document.scrollingElement) {
      window.scrollTo({ top: y, behavior: "smooth" });
    } else {
      container.scrollTo({ top: y, behavior: "smooth" });
    }
  };

  return (
    <Box>
    <div className="docs-layout">
      <aside className="docs-sidebar">
        <DocsSidebar toc={toc} activeId={activeId} onLinkClick={handleLinkClick} />
      </aside>

      {/* This may or may not be the scroll container — code above will auto-detect */}
      <main className="docs-content">
        <div className="docs-article" ref={articleRef}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkSlug]}
            rehypePlugins={[[rehypeAutolinkHeadings, { behavior: "wrap" }]]}
          >
            {content}
          </ReactMarkdown>
        </div>
      </main>
    </div>
    </Box>
  );
}
