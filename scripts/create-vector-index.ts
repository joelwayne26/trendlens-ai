/**
 * Create Vector Search Index on MongoDB Atlas
 * Run: npx tsx scripts/create-vector-index.ts
 */

import { MongoClient, Db } from 'mongodb';

const MONGO_URI = process.env.MONGO_URI || 'mongodb+srv://012240402_db_user:qQOpS7exDHBxFXiY@posts.a8pbuqs.mongodb.net/';
const DB_NAME = process.env.MONGO_DB_NAME || 'trendlens';

async function createIndexViaCommand(db: Db) {
  console.log('Using db.command() method...');
  const result = await db.command({
    createSearchIndexes: 'embeddings',
    indexes: [
      {
        name: 'vector_index',
        type: 'vectorSearch',
        definition: {
          fields: [
            { type: 'vector', path: 'embedding', numDimensions: 384, similarity: 'cosine' },
            { type: 'filter', path: 'category' },
            { type: 'filter', path: 'engagement_rate' },
          ],
        },
      },
    ],
  });
  return result;
}

async function main() {
  console.log('Connecting to MongoDB Atlas...');
  const client = new MongoClient(MONGO_URI, {
    serverSelectionTimeoutMS: 10000,
    connectTimeoutMS: 10000,
  });

  try {
    await client.connect();
    console.log('Connected!');

    const db = client.db(DB_NAME);

    // Check if collection exists, create it if not
    const collections = await db.listCollections({ name: 'embeddings' }).toArray();
    if (collections.length === 0) {
      console.log('Creating "embeddings" collection...');
      await db.createCollection('embeddings');
    }

    console.log('Creating Vector Search Index "vector_index"...');
    console.log('');

    try {
      const result = await createIndexViaCommand(db);
      console.log('SUCCESS! Vector Search Index created!');
      console.log('Result:', JSON.stringify(result, null, 2));
    } catch (error: any) {
      if (error?.message?.includes('already exists')) {
        console.log('Index "vector_index" already exists! No action needed.');
      } else {
        console.error('ERROR:', error.message);
        console.error('');
        console.error('Trying createSearchIndex helper method...');
        try {
          const collection = db.collection('embeddings');
          const idxResult = await (collection as any).createSearchIndex({
            name: 'vector_index',
            type: 'vectorSearch',
            definition: {
              fields: [
                { type: 'vector', path: 'embedding', numDimensions: 384, similarity: 'cosine' },
                { type: 'filter', path: 'category' },
                { type: 'filter', path: 'engagement_rate' },
              ],
            },
          });
          console.log('SUCCESS via helper method!', idxResult);
        } catch (err2: any) {
          if (err2?.message?.includes('already exists')) {
            console.log('Index "vector_index" already exists!');
          } else {
            console.error('Helper method also failed:', err2.message);
          }
        }
      }
    }

    console.log('');
    console.log('The index will be "Building" for 1-3 minutes, then "Active".');
    console.log('Test RAG at: https://trendlens-ai-weld.vercel.app/api/rag/search?caption=wedding+cake&category=cake&limit=3');

  } catch (error: any) {
    console.error('Connection error:', error.message);
  } finally {
    await client.close();
    console.log('Done.');
  }
}

main();
