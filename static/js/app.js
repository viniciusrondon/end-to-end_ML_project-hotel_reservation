/**
 * ET-287 Hotel Reservation ML — UI interactions
 */
(function () {
  "use strict";

  const pipelineSteps = [
    {
      title: "Data Ingestion",
      detail:
        "Reservation records are acquired from the Kaggle Hotel Reservations dataset and partitioned into training and test sets, establishing the raw signal source for downstream learning.",
    },
    {
      title: "Preprocessing",
      detail:
        "Categorical encodings, skewness treatment, correlation analysis, and SMOTE balancing transform heterogeneous booking attributes into a consistent feature space.",
    },
    {
      title: "Feature Engineering",
      detail:
        "Variance inflation factor (VIF) screening and feature selection reduce redundancy—analogous to dimensionality control in neural signal pipelines.",
    },
    {
      title: "Model Training",
      detail:
        "A LightGBM classifier is tuned via randomized search and tracked with MLflow, optimizing decision boundaries for cancellation vs. honored bookings.",
    },
    {
      title: "Deployment",
      detail:
        "The serialized model powers this inference interface, mapping live reservation inputs to class probabilities in real time.",
    },
  ];

  const fieldHints = {
    lead_time: "Days between booking and arrival—strong predictor of cancellation risk.",
    no_of_special_requests: "Count of special requests; may reflect guest engagement.",
    avg_price_per_room: "Average room rate (currency units) for the stay.",
    arrival_month: "Scheduled month of arrival (1–12).",
    arrival_date: "Day of month for scheduled arrival (1–31).",
    market_segment_type: "Distribution channel: Aviation, Complementary, Corporate, Offline, or Online.",
    no_of_week_nights: "Number of weekday nights in the reservation.",
    no_of_weekend_nights: "Number of weekend nights in the reservation.",
    type_of_meal_plan: "Selected meal plan category (encoded).",
    room_type_reserved: "Reserved room category (encoded).",
  };

  function initScrollReveal() {
    const sections = document.querySelectorAll(".section");
    if (!sections.length || !("IntersectionObserver" in window)) {
      sections.forEach((s) => s.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
          }
        });
      },
      { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
    );
    sections.forEach((s) => observer.observe(s));
  }

  function initPipeline() {
    const steps = document.querySelectorAll(".pipeline-step");
    const detailEl = document.getElementById("pipeline-detail");
    if (!steps.length || !detailEl) return;

    function activate(index) {
      steps.forEach((step, i) => {
        step.classList.toggle("is-active", i === index);
      });
      detailEl.textContent = pipelineSteps[index].detail;
    }

    steps.forEach((step, index) => {
      step.addEventListener("mouseenter", () => activate(index));
      step.addEventListener("focus", () => activate(index));
      step.addEventListener("click", () => activate(index));
    });

    activate(0);
  }

  function initFieldHints() {
    Object.entries(fieldHints).forEach(([id, text]) => {
      const input = document.getElementById(id);
      const hint = document.querySelector(`[data-hint-for="${id}"]`);
      if (!input || !hint) return;
      input.addEventListener("focus", () => {
        hint.textContent = text;
      });
      input.addEventListener("blur", () => {
        hint.textContent = "";
      });
    });
  }

  function initSignalBars() {
    const bars = document.querySelectorAll(".signal-bar");
    bars.forEach((bar) => {
      const h = 20 + Math.random() * 80;
      bar.style.height = `${h}%`;
    });
  }

  function initNavHighlight() {
    const navLinks = document.querySelectorAll(".site-nav a");
    const sections = [...document.querySelectorAll("section[id]")];
    if (!navLinks.length || !sections.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = entry.target.getAttribute("id");
            navLinks.forEach((link) => {
              link.style.fontWeight = link.getAttribute("href") === `#${id}` ? "700" : "600";
            });
          }
        });
      },
      { threshold: 0.4 }
    );
    sections.forEach((s) => observer.observe(s));
  }

  document.addEventListener("DOMContentLoaded", () => {
    initScrollReveal();
    initPipeline();
    initFieldHints();
    initSignalBars();
    initNavHighlight();
  });
})();
