import { useEffect, useState } from "react";
import "./index.css";

export default function ImageFromPath({ relativePath, imgStyle, ...imgProps }) {
    const [imgData, setImgData] = useState(null);

    useEffect(() => {
        let mounted = true;

        window.api.call("getImageData", relativePath).then((res) => {
            if (!mounted) return;
            if (res.success) setImgData(res.content);
            else console.error(`Error loading image ${relativePath}: ${res.error}`);
        });

        return () => {
            mounted = false;
        };
    }, [relativePath]);

    return (
        <div className="img-wrapper">
            {imgData ? (
                <img
                    src={imgData}
                    alt={relativePath}
                    className="img-main"
                    style={imgStyle}
                    {...imgProps}
                />
            ) : (
                <div className="img-loading-box">Loading...</div>
            )}
        </div>
    );
}
