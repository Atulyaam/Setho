/**
 * Internationalization (i18n) Dictionary for Stetho Industrial Intelligence.
 * Supports English (en) and Hindi (hi) for multi-lingual industrial field operations.
 */

export type Language = "en" | "hi";

export interface Translations {
  [key: string]: {
    en: string;
    hi: string;
  };
}

export const translations: Translations = {
  // Navigation & General
  appName: { en: "Stetho Intelligence", hi: "स्टेथो इंटेलिजेंस" },
  commandCenter: { en: "Command Center", hi: "कमांड सेंटर" },
  machines: { en: "Machine Intelligence", hi: "मशीन इंटेलिजेंस" },
  acousticStudio: { en: "Acoustic Studio", hi: "ध्वनि स्टूडियो" },
  monitoring: { en: "Fleet Priority", hi: "फ्लीट प्राथमिकता" },
  maintenance: { en: "Maintenance Assistant", hi: "रखरखाव सहायक" },
  settings: { en: "Settings", hi: "सेटिंग्स" },
  logout: { en: "Sign Out", hi: "साइन आउट" },
  login: { en: "Sign In", hi: "साइन इन" },
  register: { en: "Register", hi: "पंजीकरण" },
  systemOnline: { en: "Telemetry Online", hi: "टेलीमेट्री ऑनलाइन" },
  offlineMode: { en: "Offline Queue Active", hi: "ऑफ़लाइन कतार सक्रिय" },

  // Command Center
  fleetOverview: { en: "Industrial Acoustic Fleet Intelligence", hi: "औद्योगिक ध्वनिक फ्लीट इंटेलिजेंस" },
  totalMachines: { en: "Monitored Assets", hi: "निगरानी की गई मशीनें" },
  normalStatus: { en: "Normal-Like Acoustic", hi: "सामान्य ध्वनिक स्थिति" },
  unusualStatus: { en: "Unusual Deviations", hi: "असामान्य विचलन" },
  inconclusiveStatus: { en: "Inconclusive / Retest", hi: "अनिर्णायक / पुनः परीक्षण" },
  fleetPriorityTitle: { en: "Fleet Inspection Priority", hi: "फ्लीट निरीक्षण प्राथमिकता" },
  recentInspections: { en: "Recent Acoustic Inspections", hi: "हाल के ध्वनिक निरीक्षण" },
  noRecentInspections: { en: "No inspections recorded yet", hi: "अभी तक कोई निरीक्षण दर्ज नहीं किया गया" },

  // Acoustic Studio & Inspection
  acousticStudioSubtitle: {
    en: "High-fidelity audio acquisition, live microphone telemetry, and acoustic signature analysis.",
    hi: "उच्च-सटीक ऑडियो रिकॉर्डिंग, लाइव माइक्रोफोन टेलीमेट्री और ध्वनिक सिग्नेचर विश्लेषण।",
  },
  recordLiveAudio: { en: "Live Microphone Capture", hi: "लाइव माइक्रोफ़ोन रिकॉर्डिंग" },
  uploadAudioFile: { en: "Upload Acoustic WAV", hi: "ध्वनि WAV फ़ाइल अपलोड करें" },
  startRecording: { en: "Start Recording", hi: "रिकॉर्डिंग शुरू करें" },
  stopRecording: { en: "Stop & Analyze", hi: "रोकें और विश्लेषण करें" },
  recordingNotice: { en: "Recording in progress... Maintain constant distance.", hi: "रिकॉर्डिंग जारी है... स्थिर दूरी बनाए रखें।" },
  distanceGuidance: {
    en: "Hold microphone 15-25 cm from bearing housing. Keep steady to minimize handling rattle.",
    hi: "माइक्रोफोन को बेयरिंग हाउसिंग से 15-25 सेमी दूर रखें। हाथ की खड़खड़ाहट कम करने के लिए स्थिर रहें।",
  },

  // Decision & Badges
  normalLikeDecision: { en: "NORMAL-LIKE", hi: "सामान्य" },
  unusualDecision: { en: "UNUSUAL DEVIATION", hi: "असामान्य विचलन" },
  inconclusiveDecision: { en: "INCONCLUSIVE", hi: "अनिर्णायक" },
  provisionalBadge: { en: "PROVISIONAL CALIBRATION", hi: "अस्थायी अंशांकन" },
  calibratedBadge: { en: "ENTERPRISE CALIBRATED", hi: "पूर्ण अंशांकित" },

  // Maintenance Assistant
  maintenanceTitle: { en: "Evidence-Based Maintenance Guidance", hi: "साक्ष्य-आधारित रखरखाव मार्गदर्शन" },
  maintenanceSubtitle: {
    en: "Deterministic inspection checklists linked directly to empirical acoustic band shifts.",
    hi: "अनुभवजन्य ध्वनिक आवृत्ति बदलावों से सीधे जुड़े निरीक्षण चेकलिस्ट।",
  },
  urgencyHigh: { en: "HIGH PRIORITY INSPECTION", hi: "उच्च प्राथमिकता निरीक्षण" },
  urgencyMedium: { en: "EVALUATION RECOMMENDED", hi: "मूल्यांकन की सिफारिश" },
  urgencyLow: { en: "ROUTINE MONITORING", hi: "नियमित निगरानी" },
  urgencyVerify: { en: "REPEAT RECORDING NEEDED", hi: "पुनः रिकॉर्डिंग आवश्यक" },

  // Features
  acousticFingerprint: { en: "Acoustic Fingerprint Reference", hi: "ध्वनिक फिंगरप्रिंट संदर्भ" },
  driftTimeline: { en: "Acoustic Drift Over Time", hi: "समय के साथ ध्वनिक बहाव (ड्रिफ्ट)" },
  qrIdentity: { en: "QR Asset Identity", hi: "क्यूआर मशीन पहचान" },
  scanQr: { en: "Scan Machine QR", hi: "मशीन क्यूआर स्कैन करें" },
  offlineQueueTitle: { en: "Offline Inspection Sync", hi: "ऑफ़लाइन निरीक्षण सिंक" },
  syncNow: { en: "Sync Pending Inspections", hi: "लंबित निरीक्षण सिंक करें" },

  // Settings
  themeToggle: { en: "Interface Theme", hi: "इंटरफ़ेस थीम" },
  darkTheme: { en: "Industrial Dark", hi: "डार्क मोड" },
  lightTheme: { en: "Clean Light", hi: "लाइट मोड" },
  languageSelect: { en: "Language / भाषा", hi: "भाषा / Language" },
};

export function getTranslation(key: string, lang: Language): string {
  if (translations[key] && translations[key][lang]) {
    return translations[key][lang];
  }
  return key;
}

