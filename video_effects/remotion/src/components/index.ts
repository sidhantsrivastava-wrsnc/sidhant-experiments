import React from "react";
import { AnimatedTitle } from "./AnimatedTitle";
import { LowerThird } from "./LowerThird";
import { Listicle } from "./Listicle";
import { DataAnimation } from "./DataAnimation";
import { Subtitles } from "./Subtitles";

type ComponentMap = {
  [key: string]: React.FC<any>;
};

export const ComponentRegistry: ComponentMap = {
  animated_title: AnimatedTitle as React.FC<any>,
  lower_third: LowerThird as React.FC<any>,
  listicle: Listicle as React.FC<any>,
  data_animation: DataAnimation as React.FC<any>,
  subtitles: Subtitles as React.FC<any>,
};

// Merge generated infographic components (if any exist)
try {
  const { GeneratedRegistry } = require("./generated/_registry");
  Object.assign(ComponentRegistry, GeneratedRegistry);
} catch {
  // No generated components — fine
}
