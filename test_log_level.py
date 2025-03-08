import logging
import kompot

# Print the current log level 
print(f"Kompot logger level: {kompot.logger.level}")
print(f"Level name: {logging.getLevelName(kompot.logger.level)}")

# Check handler configuration
print("\nHandler configuration:")
for handler in kompot.logger.handlers:
    print(f"Handler: {handler}")
    print(f"Handler level: {handler.level} ({logging.getLevelName(handler.level)})")
    
print("\nTesting log messages at different levels:")
kompot.logger.debug("This is a DEBUG message")
kompot.logger.info("This is an INFO message")
kompot.logger.warning("This is a WARNING message")
kompot.logger.error("This is an ERROR message")
