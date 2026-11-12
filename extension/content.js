// SentiPulse Content Script: Extract review text from e-commerce and feedback pages

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "SCRAPE_REVIEWS") {
    const reviews = extractReviewsFromPage();
    sendResponse({ reviews });
  }
  return true; // Keep message channel open for async response
});

function extractReviewsFromPage() {
  const reviews = [];

  // 1. Amazon Product Reviews
  const amazonSelectors = [
    '[data-hook="review-body"] span',
    '[data-hook="review-collapsed"] span',
    '.review-text-content span'
  ];
  for (const selector of amazonSelectors) {
    document.querySelectorAll(selector).forEach((el) => {
      const text = el.innerText.trim();
      if (text.length > 15 && !reviews.includes(text)) {
        reviews.push(text);
      }
    });
  }

  // 2. Trustpilot Reviews
  const trustpilotSelectors = [
    '[data-service-review-text-typography="true"]',
    '.typography_body-l__KUYFJ',
    'p[data-service-review-rating-typography]'
  ];
  for (const selector of trustpilotSelectors) {
    document.querySelectorAll(selector).forEach((el) => {
      const text = el.innerText.trim();
      if (text.length > 15 && !reviews.includes(text)) {
        reviews.push(text);
      }
    });
  }

  // 3. Google Reviews / Maps
  document.querySelectorAll(".wiI7pd").forEach((el) => {
    const text = el.innerText.trim();
    if (text.length > 15 && !reviews.includes(text)) {
      reviews.push(text);
    }
  });

  // 4. Generic Fallback: Search for elements with review-like classes or data attributes
  if (reviews.length === 0) {
    const genericElements = document.querySelectorAll(
      '[class*="review" i], [class*="comment" i], [class*="testimonial" i], [class*="feedback" i]'
    );
    genericElements.forEach((el) => {
      // Avoid scraping entire containers or headers
      if (el.children.length === 0 || el.querySelectorAll("p").length === 0) {
        const text = el.innerText.trim();
        if (text.length > 20 && text.length < 1500 && !reviews.includes(text)) {
          reviews.push(text);
        }
      }
    });
  }

  return reviews.slice(0, 50); // Cap at 50 reviews
}
