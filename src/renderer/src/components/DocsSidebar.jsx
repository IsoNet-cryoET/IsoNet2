import React from "react";

export default function DocsSidebar({ toc, activeId, onLinkClick }) {
  return (
    <div>
      <nav className="docs-toc">
        {toc.map((item) => (
          <a
            key={item.id || `${item.text}-${item.level}`}
            href={`#${item.id}`}
            className={[
              "docs-toc-item",
              `level-${item.level}`,
              activeId === item.id ? "active" : ""
            ].join(" ")}
            onClick={(e) => {
              e.preventDefault();
              if (item.id) onLinkClick(item.id);
            }}
            title={item.text}
          >
            {item.text}
          </a>
        ))}
      </nav>
    </div>
  );
}
