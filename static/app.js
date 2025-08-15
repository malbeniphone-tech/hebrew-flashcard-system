// Global click handler for flashcards.  Uses event delegation to avoid
// registering multiple handlers and prevents double toggles on rapid clicks.
document.addEventListener("click", (e) => {
  const card = e.target.closest(".flashcard");
  if (!card) return;
  // simple lock to avoid duplicate toggles due to bubbling or double clicks
  if (card.dataset.busy === "1") return;
  card.dataset.busy = "1";
  Promise.resolve().then(() => delete card.dataset.busy);
  card.classList.toggle("revealed");
}, { passive: true });
