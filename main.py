"""Entry point for the pytorch-quality-bench application."""

import logging

logger = logging.getLogger(__name__)


def main():
  """Run the application's main entry point."""
  logger.info("Hello from pytorch-quality-bench!")


if __name__ == "__main__":
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  main()
