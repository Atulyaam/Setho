"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { Language, getTranslation } from "../lib/i18n";

type Theme = "dark" | "light";

interface UIContextType {
  theme: Theme;
  toggleTheme: () => void;
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
  isOnline: boolean;
}

const UIContext = createContext<UIContextType | undefined>(undefined);

export function UIProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>("dark");
  const [language, setLanguageState] = useState<Language>("en");
  const [isOnline, setIsOnline] = useState<boolean>(true);

  useEffect(() => {
    // Check saved theme
    const savedTheme = localStorage.getItem("stetho_theme") as Theme | null;
    if (savedTheme) {
      setTheme(savedTheme);
      document.documentElement.classList.toggle("light-theme", savedTheme === "light");
    }

    // Check saved language
    const savedLang = localStorage.getItem("stetho_lang") as Language | null;
    if (savedLang) {
      setLanguageState(savedLang);
    }

    // Online / Offline status
    if (typeof window !== "undefined") {
      setIsOnline(navigator.onLine);
      const handleOnline = () => setIsOnline(true);
      const handleOffline = () => setIsOnline(false);

      window.addEventListener("online", handleOnline);
      window.addEventListener("offline", handleOffline);

      return () => {
        window.removeEventListener("online", handleOnline);
        window.removeEventListener("offline", handleOffline);
      };
    }
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    localStorage.setItem("stetho_theme", nextTheme);
    document.documentElement.classList.toggle("light-theme", nextTheme === "light");
  };

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
    localStorage.setItem("stetho_lang", lang);
  };

  const t = (key: string) => getTranslation(key, language);

  return (
    <UIContext.Provider
      value={{
        theme,
        toggleTheme,
        language,
        setLanguage,
        t,
        isOnline,
      }}
    >
      {children}
    </UIContext.Provider>
  );
}

export function useUI() {
  const context = useContext(UIContext);
  if (!context) {
    throw new Error("useUI must be used within a UIProvider");
  }
  return context;
}

