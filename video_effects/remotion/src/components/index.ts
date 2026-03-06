import React from "react";
import { AnimatedTitle } from "./AnimatedTitle";
import type { AnimatedTitleProps } from "../types";

type ComponentMap = {
  [key: string]: React.FC<any>;
};

export const ComponentRegistry: ComponentMap = {
  animated_title: AnimatedTitle as React.FC<any>,
};
